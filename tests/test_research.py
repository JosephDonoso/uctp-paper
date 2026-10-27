"""Metrics, budget, matching, raw-data and general-analysis checks."""

import contextlib
import io
import json
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from uctp.experiments.analysis import analyze
from uctp.experiments.cli import build_parser, main
from uctp.experiments.comparisons import compare_factors
from uctp.experiments.records import read_records, write_table
from uctp.experiments.runner import run_grid
from uctp.experiments.statistics import holm, independent_difference, sign_probability
from uctp.instance_io import parse_instance
from uctp.metrics import measure_solution
from uctp.objective import SearchObjective
from uctp.planner import Approach, SolveOptions, solve
from uctp.reporting import execute_run
from uctp.sectioning import StudentSectioning

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/small.json"


def fake_records():
    rows = []
    for instance in range(3):
        for seed in [1, 2]:
            for treatment, value in [("A", seed), ("B", seed + 2)]:
                rows.append(
                    dict(
                        run_id=f"{instance}-{seed}-{treatment}",
                        input_sha256=str(instance),
                        seed=seed,
                        approach=treatment,
                        group="BASE",
                        alpha=1,
                        status="solved",
                        F_ref=value,
                    )
                )
    return rows


def contrasts(rows, unit="instance", metrics=None):
    return compare_factors(
        rows,
        ["approach"],
        metrics or ["F_ref"],
        mode="paired",
        pair_by=["input_sha256", "seed"],
        unit=unit,
        instance_field="input_sha256",
        strata=None,
        rng=np.random.default_rng(7),
        replicates=200,
    )


class ResearchTests(unittest.TestCase):
    def test_metrics_match_objective_and_detail_totals_without_changing_rng(self):
        instance = parse_instance(
            json.loads(EXAMPLE.read_text()), max_daily_load=4, max_block_load=15
        )
        for alpha in [0, 0.25, 1, 2]:
            result = solve(instance, SolveOptions(seed=7, alpha=alpha, iterations_per_search=2))
            state = random.getstate()
            report = measure_solution(instance, result.solution, alpha=alpha, details=True)
            self.assertEqual(random.getstate(), state)
            self.assertAlmostEqual(
                report["metrics"]["F"],
                SearchObjective(instance, alpha=alpha, cv_tolerance=0.1, idle_tolerance=4).evaluate(
                    result.solution
                ),
            )
            self.assertAlmostEqual(
                report["metrics"]["F_ref"], report["metrics"]["R_imb"] * report["metrics"]["R_dis"]
            )
            self.assertEqual(
                report["metrics"]["conflicts"],
                sum(r["conflicts"] for r in report["details"]["students"]),
            )
            self.assertEqual(
                report["metrics"]["dissatisfaction_total"],
                sum(r["inconvenience_cost"] for r in report["details"]["students"]),
            )
            self.assertEqual(
                report["metrics"]["assigned_courses"],
                sum(r["enrollment"] for r in report["details"]["sections"]),
            )

    def test_recording_preserves_seeded_solution_and_random_stream(self):
        instance = parse_instance(
            json.loads(EXAMPLE.read_text()), max_daily_load=4, max_block_load=15
        )
        options = SolveOptions(seed=42, iterations_per_search=3)
        plain = solve(instance, options)
        plain_state = random.getstate()
        measured = solve(instance, options, collect_metrics=True)
        self.assertEqual(plain.solution, measured.solution)
        self.assertEqual(plain_state, random.getstate())
        self.assertIsNotNone(measured.initial_solution)
        self.assertEqual(measured.diagnostics["completed_iterations"], 3)
        traced = solve(instance, replace(options, trace=True), collect_metrics=True)
        self.assertEqual(traced.solution, plain.solution)
        self.assertEqual(len(traced.trace), 3)
        self.assertEqual([r["iteration"] for r in traced.trace], [1, 2, 3])

    def test_ablation_does_not_call_local_refinement(self):
        instance = parse_instance(
            json.loads(EXAMPLE.read_text()), max_daily_load=4, max_block_load=15
        )
        with patch.object(
            StudentSectioning, "_refine_sections", side_effect=AssertionError("LS called")
        ):
            result = solve(
                instance, SolveOptions(seed=1, local_search=False, iterations_per_search=1)
            )
        self.assertTrue(result.solution.Y)

    def test_total_iteration_cap_in_last_decoupled_block(self):
        instance = parse_instance(
            json.loads(EXAMPLE.read_text()), max_daily_load=4, max_block_load=15
        )
        result = solve(
            instance,
            SolveOptions(
                seed=1,
                approach=Approach.DECOUPLED,
                iterations_per_search=2,
                search_blocks=5,
                total_iteration_limit=3,
            ),
        )
        self.assertEqual(result.diagnostics["completed_iterations"], 3)
        self.assertEqual(result.diagnostics["search_blocks_completed"], 2)

    def test_failure_record_preserves_attempts_and_null_quality(self):
        with tempfile.TemporaryDirectory() as folder:
            data = json.loads(EXAMPLE.read_text())
            data["teachers"][0]["available_blocks"] = []
            path = Path(folder) / "impossible.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            record = execute_run(path, SolveOptions(seed=1, max_retries=0))
            self.assertEqual(record["status"], "construction_failed")
            self.assertIsNone(record["metrics"])
            self.assertEqual(record["diagnostics"]["construction_attempts"], 1)
            self.assertEqual(record["X"], [])
            output = Path(folder) / "analysis"
            inputs = Path(folder) / "failed.json"
            inputs.write_text(json.dumps(record), encoding="utf-8")
            result = analyze(
                [inputs],
                output,
                metrics=["F_ref"],
                group_by=["group"],
                filters=[],
                compare=[],
                mode="paired",
                pair_by=["input_sha256", "seed"],
                unit="instance",
                instance_field="input_sha256",
                strata=None,
                replicates=100,
                seed=7,
            )
            self.assertEqual(result["success_rates"][0]["failed"], 1)
            self.assertIsNone(result["summaries"][0]["mean"])

    def test_instance_units_do_not_treat_seeds_as_independent_instances(self):
        quality, reliability, pairs = contrasts(fake_records())
        self.assertEqual(quality[0]["left_units"], 3)
        self.assertEqual(quality[0]["matched_successful_runs"], 6)
        self.assertEqual(quality[0]["mean_difference"], -2)
        self.assertEqual(quality[0]["p"], 0.25)
        self.assertEqual(len(pairs), 6)
        self.assertEqual(reliability[0]["matched_units"], 3)
        run_quality, _, _ = contrasts(fake_records(), unit="run")
        self.assertEqual(run_quality[0]["left_units"], 6)
        self.assertEqual(run_quality[0]["p"], 0.03125)
        mixed = [
            dict(row, alpha=1.0, seed=float(row["seed"])) if row["approach"] == "B" else row
            for row in fake_records()
        ]
        self.assertEqual(contrasts(mixed)[0][0]["matched_successful_runs"], 6)

    def test_duplicate_pair_keys_and_cross_weight_native_F_are_rejected(self):
        rows = fake_records()
        with self.assertRaises(ValueError):
            contrasts(rows + [rows[0].copy()])
        rows = [dict(r, F=1) for r in fake_records()]
        with self.assertRaises(ValueError):
            compare_factors(
                rows,
                ["alpha"],
                ["F"],
                mode="paired",
                pair_by=["input_sha256", "seed"],
                unit="instance",
                instance_field="input_sha256",
                strata=None,
                rng=np.random.default_rng(7),
                replicates=100,
            )

    def test_failed_pairs_do_not_receive_artificial_metric_values(self):
        rows = fake_records()
        for row in rows:
            if row["input_sha256"] == "0" and row["approach"] == "A":
                row["status"] = "construction_failed"
                row["F_ref"] = None
        quality, reliability, _ = contrasts(rows)
        self.assertEqual(quality[0]["left_units"], 2)
        self.assertEqual(reliability[0]["right_only_success"], 1)

    def test_exact_statistics_and_small_sample_behavior(self):
        self.assertEqual(sign_probability(0, 10), 2 / 1024)
        self.assertEqual(sign_probability(0, 0), 1)
        rows = [{"p": 0.01}, {"p": 0.04}, {"p": 0.03}, {"p": None}]
        holm(rows)
        np.testing.assert_allclose([r["p_holm"] for r in rows[:3]], [0.03, 0.06, 0.06])
        independent = independent_difference([1, 2], [5, 6], np.random.default_rng(7), 200)
        self.assertEqual(independent["mean_difference"], -4)
        self.assertAlmostEqual(independent["p"], 1 / 3)

    def test_grid_resume_and_analysis_csv_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs = root / "BASE"
            inputs.mkdir()
            data = json.loads(EXAMPLE.read_text())
            for i in range(2):
                modified = json.loads(json.dumps(data))
                modified["students"][0]["id"] = f"instance-{i}-student"
                (inputs / f"instance_{i}.json").write_text(json.dumps(modified), encoding="utf-8")
            argv = [
                "run",
                "--inputs",
                str(inputs),
                "--output",
                str(root / "runs"),
                "--seeds",
                "1",
                "2",
                "--approaches",
                "Embedded",
                "Decoupled-2",
                "--local-search",
                "on",
                "off",
                "--iteration-budgets",
                "3",
                "--max-retries",
                "5",
                "--details",
                "--save-solutions",
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                args = build_parser().parse_args(argv)
                manifest = run_grid(args)
                self.assertEqual(manifest["planned_runs"], 16)
                with patch(
                    "uctp.experiments.runner.execute_run",
                    side_effect=AssertionError("Repeated run"),
                ):
                    run_grid(build_parser().parse_args(argv + ["--resume"]))
                self.assertEqual(
                    main(
                        [
                            "analyze",
                            "--inputs",
                            str(root / "runs"),
                            "--output",
                            str(root / "tables"),
                            "--compare",
                            "approach",
                            "local_search",
                            "--bootstrap",
                            "100",
                        ]
                    ),
                    0,
                )
            records = read_records([root / "runs"])
            self.assertEqual(len(records), 16)

            def check_exported_keys(value):
                if isinstance(value, dict):
                    self.assertTrue(all("tabu" not in key.lower() for key in value))
                    for child in value.values():
                        check_exported_keys(child)
                elif isinstance(value, list):
                    for child in value:
                        check_exported_keys(child)

            for record in records:
                check_exported_keys(record)
            self.assertNotIn(
                "tabu", (root / "runs/raw.csv").read_text(encoding="utf-8").splitlines()[0].lower()
            )
            self.assertTrue(all(r["diagnostics"]["completed_iterations"] <= 3 for r in records))
            self.assertTrue((root / "tables/students.csv").is_file())
            self.assertTrue((root / "tables/sections.csv").is_file())
            table = json.loads((root / "tables/analysis.json").read_text())
            self.assertGreater(len(table["contrasts"]), 0)
            merged = read_records([root / "runs", root / "runs/raw.csv"])
            self.assertEqual(len(merged), 16)
            self.assertTrue(all("details" in record for record in merged))

    def test_filters_and_independent_custom_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "custom.csv"
            write_table(path, [dict(row, alpha=1.0) for row in fake_records()])
            result = analyze(
                [path],
                root / "tables",
                metrics=["F_ref"],
                group_by=["group", "approach"],
                filters=["group=BASE", "alpha=1"],
                compare=["approach"],
                mode="independent",
                pair_by=[],
                unit="instance",
                instance_field="input_sha256",
                strata=None,
                replicates=100,
                seed=7,
            )
            self.assertEqual(len(result["contrasts"]), 1)
            self.assertEqual(result["contrasts"][0]["mean_difference"], -2)

    def test_budget_contrast_does_not_hold_its_derived_iteration_cap_fixed(self):
        rows = [
            dict(
                row,
                budget_kind="iterations",
                budget=budget,
                iteration_budget=budget,
                secondary_iteration_cap=None,
                run_id=f"{row['run_id']}-{budget}",
            )
            for row in fake_records()
            if row["approach"] == "A"
            for budget in [2, 4]
        ]
        quality, _, _ = compare_factors(
            rows,
            ["budget"],
            ["F_ref"],
            mode="paired",
            pair_by=["input_sha256", "seed"],
            unit="instance",
            instance_field="input_sha256",
            strata=None,
            rng=np.random.default_rng(7),
            replicates=100,
        )
        self.assertEqual(len(quality), 1)

    def test_failed_diagnostics_can_be_described_explicitly(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [
                dict(
                    run_id=str(i),
                    input_sha256=str(i),
                    status="construction_failed",
                    seed=1,
                    construction_seconds=float(i + 1),
                )
                for i in range(3)
            ]
            path = root / "failures.csv"
            write_table(path, rows)
            result = analyze(
                [path],
                root / "tables",
                metrics=["construction_seconds"],
                group_by=[],
                filters=[],
                compare=[],
                mode="paired",
                pair_by=[],
                unit="instance",
                instance_field="input_sha256",
                strata=None,
                replicates=100,
                seed=7,
                include_failed=True,
            )
            self.assertEqual(result["summaries"][0]["mean"], 2)

    def test_run_identifiers_distinguish_retry_budgets(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            identifiers = []
            for retries in [1, 2]:
                output = root / f"retry_{retries}"
                args = build_parser().parse_args(
                    [
                        "run",
                        "--inputs",
                        str(EXAMPLE),
                        "--output",
                        str(output),
                        "--approaches",
                        "Embedded",
                        "--iteration-budgets",
                        "1",
                        "--seeds",
                        "703",
                        "--max-retries",
                        str(retries),
                    ]
                )
                run_grid(args)
                identifiers.append(next((output / "records").glob("*.json")).name)
            self.assertNotEqual(identifiers[0], identifiers[1])

    def test_default_summaries_keep_custom_research_tags_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "cohorts.csv"
            write_table(
                path,
                [
                    dict(row, run_id=row["run_id"] + cohort, tag_cohort=cohort)
                    for row in fake_records()
                    for cohort in ["first", "second"]
                ],
            )
            result = analyze(
                [path],
                root / "tables",
                metrics=["F_ref"],
                group_by=None,
                filters=[],
                compare=[],
                mode="paired",
                pair_by=[],
                unit="instance",
                instance_field="input_sha256",
                strata=None,
                replicates=100,
                seed=7,
            )
            self.assertEqual(len(result["summaries"]), 4)


if __name__ == "__main__":
    unittest.main()

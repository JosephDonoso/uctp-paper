"""Checks for generator structure, reproducibility and accepted-data output."""

import contextlib
import io
import json
import random
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from uctp.generation import GenerationOptions, generate_instance
from uctp.generation.cli import main
from uctp.generation.preassignments import _preassign_course
from uctp.generation.teacher_availability import _generate_teacher_availability
from uctp.generation.sampling import _sample_counts
from uctp.instance_io import load_instance, parse_instance
from uctp.legacy_input import convert_legacy_instance


class GenerationTests(unittest.TestCase):
    def test_availability_uses_blockwise_draws_and_nonempty_days(self):
        days = ("Lu", "Ma", "Mi", "Ju", "Vi")
        rng = random.Random(703)
        expected_rng = random.Random(703)
        expected = []
        for day in days:
            blocks = [int(expected_rng.random() < 0.7) for _ in range(7)]
            if not any(blocks):
                blocks[expected_rng.randrange(7)] = 1
            expected.append({day: blocks})
        self.assertEqual(_generate_teacher_availability(days, 0.7, rng), expected)
        for probability in (0, 1):
            result = _generate_teacher_availability(days, probability, random.Random(1))
            self.assertEqual([sum(next(iter(day.values()))) for day in result],
                             [1 if probability == 0 else 7] * 5)

    def test_students_use_group_dispersion_without_individual_sampling(self):
        from uctp.generation.student_demand import _generate_students
        import uctp.generation.student_demand as demand
        courses = [{"codigo": f"C{n}", "_level_index": n // 2,
                    "paralelos": [{"codigo": "01"}]} for n in range(12)]
        options = GenerationOptions(courses_per_level=(2, 2, 2, 2, 2, 2),
                                    student_count=8, student_load_mean=3,
                                    student_load_sd=0, curricular_sampling_sd=1.2)
        with patch.object(demand, "_sample_integer", wraps=demand._sample_integer) as sample:
            _generate_students(courses, options, random.Random(703))
        curricular_calls = [call for call in sample.call_args_list if call.args[:2] == (1, 6)]
        self.assertEqual(len(curricular_calls), 24)
        self.assertTrue(all(call.args[3] == 1.2 for call in curricular_calls))

    def test_normal_layouts_repeat_lecture_count_across_sections(self):
        data = generate_instance(GenerationOptions(student_count=5), seed=703)
        self.assertNotIn("metadatos", data)
        for course in data["eventos"]["asignaturas"]:
            sections = course["paralelos"]
            self.assertTrue(1 <= len(sections) <= 4)
            counts = {len(s["clases"]) for s in sections}
            self.assertEqual(len(counts), 1)
            self.assertTrue(2 <= next(iter(counts)) <= 4)

    def test_reference_student_count_uses_normal_parameters_not_moment_fitting(self):
        import statistics

        options = GenerationOptions()
        self.assertEqual((options.student_load_mean, options.student_load_sd), (3.2, 1.6))
        draws = _sample_counts(50000, 0, 9, 3.2, 1.6, random.Random(703))
        self.assertEqual((min(draws), max(draws)), (0, 9))
        self.assertAlmostEqual(statistics.mean(draws), 3.2, delta=0.15)
        self.assertAlmostEqual(statistics.pstdev(draws), 1.6, delta=0.2)

    def test_custom_normal_parameters_validate_without_final_moment_constraints(self):
        for mean in (0, 3.3, 9):
            GenerationOptions(student_load_mean=mean, student_load_sd=0).validate()
        for mean, sd in ((-1, 1), (10, 1), (3, -1), (float("nan"), 1), (3, float("inf"))):
            with self.assertRaises(ValueError):
                GenerationOptions(student_load_mean=mean, student_load_sd=sd).validate()

    def test_zero_load_students_keep_empty_course_requests(self):
        data = generate_instance(
            GenerationOptions(student_count=20, student_load_mean=0, student_load_sd=0), seed=703
        )
        self.assertEqual(len(data["recursos"]["estudiantes"]), 20)
        self.assertTrue(all(not s["asignaturas"] for s in data["recursos"]["estudiantes"]))

    def test_cli_applies_custom_student_load_without_metadata(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(
                main(
                    [
                        "--output",
                        folder,
                        "--count",
                        "1",
                        "--students",
                        "20",
                        "--teacher-availability",
                        "1",
                        "--preassignment-probability",
                        "0",
                        "--seed",
                        "703",
                        "--max-candidates",
                        "10",
                        "--student-load-mean",
                        "0",
                        "--student-load-sd",
                        "0",
                    ]
                ),
                0,
            )
            data = json.loads((Path(folder) / "BASE/instance_001.json").read_text(encoding="utf-8"))
            self.assertNotIn("metadatos", data)
            self.assertTrue(all(not student["asignaturas"] for student in data["recursos"]["estudiantes"]))

    def test_section_staffing_and_preassignments(self):
        sizes = Counter()
        for seed in range(40):
            options = GenerationOptions(student_count=20, preassignment_probability=1)
            data = generate_instance(options, seed=seed)
            instance = parse_instance(
                convert_legacy_instance(data), max_daily_load=data["restricciones"]["MAX_CARGA_DIARIA"], max_block_load=15
            )
            self.assertTrue(
                all(not any(key.startswith("_") for key in t) for t in data["recursos"]["maestros"])
            )
            for course in data["eventos"]["asignaturas"]:
                for section in course["paralelos"]:
                    self.assertTrue(1 <= len(section["maestros"]) <= 2)
                    self.assertEqual(len(section["maestros"]), len(set(section["maestros"])))
                    self.assertTrue(all("maestros" not in lecture for lecture in section["clases"]))
            weekly = Counter(
                teacher
                for course in data["eventos"]["asignaturas"]
                for section in course["paralelos"]
                for _ in section["clases"]
                for teacher in section["maestros"]
            )
            self.assertEqual(set(weekly), {t["nombre"] for t in data["recursos"]["maestros"]})
            self.assertTrue(all(2 <= n <= options.teacher_max_load for n in weekly.values()))
            self.assertEqual(data["restricciones"]["MAX_CARGA_DIARIA"], options.max_daily_load)
            self.assertEqual({t.max_daily_load for t in instance.teachers}, {options.max_daily_load})
            self.assertTrue(all("max_daily_load" not in t for t in data["recursos"]["maestros"]))
            fixed_days = {}
            for index, teacher in enumerate(instance.teachers):
                fixed = [
                    lecture.preassigned_block
                    for course in instance.courses
                    for section in course.sections
                    for lecture in section.lectures
                    if lecture.preassigned_block is not None and index in lecture.teacher_indices
                ]
                days = Counter(block // instance.blocks_per_day for block in fixed)
                self.assertTrue(all(n <= teacher.max_daily_load for n in days.values()))
                self.assertLessEqual(len(days), teacher.max_days)
                fixed_days[index] = days
            fixed_teacher_blocks = set()
            for course in instance.courses:
                for section in course.sections:
                    for lecture in section.lectures:
                        sizes[len(lecture.teacher_indices)] += 1
                        self.assertEqual(
                            len(lecture.teacher_indices), len(set(lecture.teacher_indices))
                        )
                        if lecture.preassigned_block is not None:
                            self.assertNotIn(lecture.preassigned_block, instance.protected_blocks)
                            for teacher in lecture.teacher_indices:
                                self.assertIn(
                                    lecture.preassigned_block,
                                    instance.teachers[teacher].available_blocks,
                                )
                                self.assertNotIn(
                                    (teacher, lecture.preassigned_block), fixed_teacher_blocks
                                )
                                fixed_teacher_blocks.add((teacher, lecture.preassigned_block))
            self.assertTrue(
                all(0 <= len(s["asignaturas"]) <= 9 for s in data["recursos"]["estudiantes"])
            )
        self.assertEqual(sizes[0], 0)
        self.assertEqual(set(sizes), {1, 2})

    def test_counts_requests_and_teacher_coverage(self):
        data = generate_instance(GenerationOptions(), seed=703)
        courses = data["eventos"]["asignaturas"]
        teachers = data["recursos"]["maestros"]
        students = data["recursos"]["estudiantes"]
        self.assertEqual((len(courses), len(students)), (30, 500))
        loads = Counter()
        for course in courses:
            self.assertTrue(1 <= len(course["paralelos"]) <= 4)
            lecture_counts = {len(section["clases"]) for section in course["paralelos"]}
            self.assertEqual(len(lecture_counts), 1)
            self.assertTrue(2 <= next(iter(lecture_counts)) <= 4)
            for section in course["paralelos"]:
                self.assertTrue(1 <= len(section["maestros"]) <= 2)
                for teacher in section["maestros"]:
                    loads[teacher] += len(section["clases"])
        self.assertEqual(set(loads), {teacher["nombre"] for teacher in teachers})
        self.assertEqual([t["nombre"] for t in teachers], [f"DOC{i:03d}" for i in range(1, len(teachers) + 1)])
        self.assertTrue(all(2 <= count <= 12 for count in loads.values()))
        for teacher in teachers:
            self.assertEqual(len(teacher["disponibilidad"]), 5)
            for day in teacher["disponibilidad"]:
                blocks = next(iter(day.values()))
                self.assertEqual(len(blocks), 7)
            self.assertTrue(any(v for day in teacher["disponibilidad"] for values in day.values() for v in values))
        for student in students:
            requests = [course["codigo"] for course in student["asignaturas"]]
            self.assertTrue(0 <= len(requests) <= 9)
            self.assertEqual(len(requests), len(set(requests)))
        instance = parse_instance(
            convert_legacy_instance(data), max_daily_load=4, max_block_load=15
        )
        self.assertEqual(len(instance.curricular_levels), 6)

    def test_seed_reproducibility_and_private_random_state(self):
        options = GenerationOptions(student_count=12)
        state = random.getstate()
        first = generate_instance(options, seed=17)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(first, generate_instance(options, seed=17))
        self.assertNotEqual(first, generate_instance(options, seed=18))

    def test_invalid_parameters_are_rejected(self):
        for field, value in [
            ("teacher_max_load", 3),
            ("student_count", 0),
            ("teacher_load_mean", 13),
            ("teacher_load_sd", -1),
            ("co_teaching_probability", 1.5),
            ("shared_teacher_probability", -0.1),
            ("max_daily_load", 0),
            ("curricular_sampling_sd", 0),
            ("level_decay", 0),
            ("section_mean", float("nan")),
            ("availability_probability", -0.1),
            ("preassignment_probability", 1.1),
        ]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                generate_instance(replace(GenerationOptions(), **{field: value}), seed=1)
        with self.assertRaises(ValueError):
            generate_instance(GenerationOptions(), seed=-1)

    def test_nearest_level_fallback_keeps_distinct_requests(self):
        options = GenerationOptions(
            courses_per_level=(9, 0, 0, 0, 0, 0), student_count=25
        )
        data = generate_instance(options, seed=3)
        for student in data["recursos"]["estudiantes"]:
            codes = [request["codigo"] for request in student["asignaturas"]]
            self.assertTrue(0 <= len(codes) <= 9)
            self.assertEqual(len(codes), len(set(codes)))

    def test_generated_global_daily_limit_and_explicit_override(self):
        data = generate_instance(GenerationOptions(student_count=5), seed=703)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "instance.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            loaded = load_instance(path, input_format="legacy", max_daily_load=None, max_block_load=15)
            self.assertEqual({t.max_daily_load for t in loaded.teachers},
                             {data["restricciones"]["MAX_CARGA_DIARIA"]})
            overridden = load_instance(path, input_format="legacy", max_daily_load=4, max_block_load=15)
            self.assertEqual({t.max_daily_load for t in overridden.teachers}, {4})
            data["recursos"]["maestros"][0]["max_daily_load"] = 0
            with self.assertRaises(ValueError):
                convert_legacy_instance(data)

    def test_failed_preassignment_rolls_back_occupancy_and_lectures(self):
        course = {"paralelos": [{"maestros": ["T"], "clases": [{}, {}]}]}
        availability = {"T": {"Lu": [1]}}
        occupied = {"T": {("Ma", "1-2")}}
        accepted = _preassign_course(
            course, ("Lu",), ("1-2",), availability, occupied, random.Random(1),
            max_daily_load=4, max_days={"T": 3},
        )
        self.assertFalse(accepted)
        self.assertEqual(occupied, {"T": {("Ma", "1-2")}})
        self.assertTrue(all(not lecture for lecture in course["paralelos"][0]["clases"]))

    def test_preassignments_respect_availability_protection_and_teacher_overlap(self):
        data = generate_instance(
            replace(GenerationOptions(), preassignment_probability=1), seed=41
        )
        availability = {
            teacher["nombre"]: {
                day: blocks for item in teacher["disponibilidad"] for day, blocks in item.items()
            }
            for teacher in data["recursos"]["maestros"]
        }
        blocks = data["tiempos"]["bloques"]
        occupied = set()
        fixed_count = 0
        for course in data["eventos"]["asignaturas"]:
            fixed_flags = []
            for section in course["paralelos"]:
                for lecture in section["clases"]:
                    fixed = lecture.get("horario_predefinido")
                    fixed_flags.append(bool(fixed))
                    if fixed:
                        fixed_count += 1
                        day, block = fixed["dia"], fixed["bloque"]
                        self.assertNotEqual((day, block), ("Ju", "7-8"))
                        for teacher in section["maestros"]:
                            self.assertEqual(availability[teacher][day][blocks.index(block)], 1)
                            key = teacher, day, block
                            self.assertNotIn(key, occupied)
                            occupied.add(key)
            self.assertTrue(all(fixed_flags) or not any(fixed_flags))
        self.assertGreater(fixed_count, 0)

    def test_cli_saves_readable_data_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            args = [
                "--output",
                folder,
                "--count",
                "1",
                "--students",
                "12",
                "--courses-per-level",
                "2",
                "2",
                "2",
                "1",
                "1",
                "1",
                "--teacher-availability",
                "1",
                "--preassignment-probability",
                "0",
                "--construction-retries",
                "2",
                "--max-candidates",
                "20",
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            destination = Path(folder) / "BASE/instance_001.json"
            contents = destination.read_bytes()
            data = json.loads(contents)
            self.assertEqual(
                set(data), {"restricciones", "tiempos", "eventos", "recursos"}
            )
            self.assertNotIn("metrics", data)
            parse_instance(convert_legacy_instance(data), max_daily_load=4, max_block_load=15)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(args)
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(destination.read_bytes(), contents)

    def test_group_keeps_institutional_limit_and_skips_rejected_candidates(self):
        from uctp.generation.cli import _generate_group
        def candidate(load):
            return {"restricciones": {"MAX_CARGA_DIARIA": 4},
                    "eventos": {"asignaturas": [{"paralelos": [
                        {"maestros": ["T"], "clases": [{} for _ in range(load)]}]}]}}
        seeds = []
        def accept(data, *, group_index, seed, retries):
            seeds.append(seed)
            return seed != 2
        with patch("uctp.generation.cli.generate_instance", side_effect=[candidate(4), candidate(9), candidate(7)]), \
             patch("uctp.generation.cli._accept_candidate", side_effect=accept), \
             contextlib.redirect_stdout(io.StringIO()):
            result = _generate_group(GenerationOptions(), group="BASE", group_index=1,
                                     count=2, seed=1, retries=0, max_candidates=3)
        self.assertEqual([seed for seed, _ in result], [1, 3])
        self.assertEqual(seeds, [1, 2, 3])
        self.assertEqual({data["restricciones"]["MAX_CARGA_DIARIA"] for _, data in result}, {4})

    def test_preassignments_respect_daily_limit_and_maximum_days(self):
        course = {"paralelos": [{"maestros": ["T"], "clases": [{} for _ in range(4)]}]}
        availability = {"T": {day: [1, 1, 1] for day in ("Lu", "Ma", "Mi")}}
        occupied = {"T": set()}
        accepted = _preassign_course(
            course, ("Lu", "Ma", "Mi"), ("1-2", "3-4", "5-6"), availability, occupied,
            random.Random(3), max_daily_load=2, max_days={"T": 2},
        )
        self.assertTrue(accepted)
        days = Counter(day for day, _ in occupied["T"])
        self.assertEqual(len(days), 2)
        self.assertTrue(all(n <= 2 for n in days.values()))

    def test_candidate_limit_records_no_unaccepted_instances(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            patch("uctp.generation.cli._accept_candidate", return_value=False) as acceptance,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(
                main(
                    ["--output", folder, "--count", "1", "--max-candidates", "2", "--students", "2"]
                ),
                1,
            )
            self.assertEqual(acceptance.call_count, 2)
            self.assertEqual(list((Path(folder) / "BASE").iterdir()), [])


if __name__ == "__main__":
    unittest.main()

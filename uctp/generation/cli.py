"""Generate accepted dataset instances using command-line parameters."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from ..constraints import InstitutionalConstraints
from ..construction import ConstructionFailure, InitialStrategy, InitialTimetabling
from ..instance_io import parse_instance
from ..legacy_input import convert_legacy_instance
from .instance_generator import generate_instance
from .parameters import GROUP_PARAMETERS, DatasetGroup, GenerationOptions


def _accept_candidate(data: dict, *, group_index: int, seed: int, retries: int) -> bool:
    instance = parse_instance(
        convert_legacy_instance(data),
        max_daily_load=data["restricciones"]["MAX_CARGA_DIARIA"],
        max_block_load=15,
    )
    constraints = InstitutionalConstraints(instance)
    if not constraints.is_feasible(constraints.preassignments):
        return False
    weekly = Counter(
        teacher
        for course in instance.courses
        for section in course.sections
        for lecture in section.lectures
        for teacher in lecture.teacher_indices
    )
    for index, teacher in enumerate(instance.teachers):
        daily = Counter(
            block // instance.blocks_per_day
            for block in teacher.available_blocks
            if block not in instance.protected_blocks
        )
        capacity = sum(
            sorted((min(teacher.max_daily_load, count) for count in daily.values()), reverse=True)[
                : teacher.max_days
            ]
        )
        if weekly[index] > capacity:
            return False
    strategies = (
        InitialStrategy.PRIORITIZATION_ONLY,
        InitialStrategy.COMPATIBILITY_MATRIX,
        InitialStrategy.RANDOM,
    )
    for index, strategy in enumerate(strategies):
        random.seed(f"{group_index}:{seed}:{index}")
        construction = InitialTimetabling(
            instance,
            constraints,
            strategy=strategy,
            max_retries=retries,
            beta=0.25,
            cv_tolerance=0.1,
        )
        try:
            solution = construction.solve()
        except ConstructionFailure:
            continue
        if constraints.is_complete(solution.X) and constraints.is_feasible(solution.X):
            return True
    return False


def _generate_group(
    options: GenerationOptions, *, group: str, group_index: int,
    count: int, seed: int, retries: int, max_candidates: int | None,
) -> list[tuple[int, dict]] | None:
    """Test consecutive candidate seeds until ``count`` instances are accepted.

    Every candidate uses the institutional daily teacher limit stored in its
    data. Rejected candidates are skipped. Only a complete group is returned.
    """
    accepted: list[tuple[int, dict]] = []
    tested = 0
    while len(accepted) < count:
        if max_candidates is not None and tested >= max_candidates:
            print(f"Candidate limit reached: {group}; current group not saved.", flush=True)
            return None
        candidate_seed = seed + tested
        tested += 1
        data = generate_instance(options, seed=candidate_seed)
        if _accept_candidate(data, group_index=group_index, seed=candidate_seed, retries=retries):
            accepted.append((candidate_seed, data))
    print(f"{group}: {count} instances accepted from {tested} candidates.", flush=True)
    return accepted


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate single-program curricular UCTP datasets."
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Parent directory for dataset groups."
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=list(DatasetGroup),
        default=[DatasetGroup.BASE],
        help="Dataset groups; default: BASE. BASE is generated once.",
    )
    parser.add_argument(
        "--count", type=int, default=50, help="Accepted instances per group; default: 50."
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="First candidate seed per group; default: 1."
    )
    parser.add_argument(
        "--construction-retries",
        type=int,
        default=20,
        help="Retries after initial construction per strategy; default: 20.",
    )
    parser.add_argument(
        "--max-candidates", type=int, help="Optional cap on candidate seeds tested per group."
    )
    parser.add_argument(
        "--teacher-load-mean",
        type=float,
        default=2.0,
        help="Normal location of teacher target weekly loads before rounding/truncation to 2-max; default: 2.0.",
    )
    parser.add_argument(
        "--teacher-load-sd",
        type=float,
        default=3.25,
        help="Normal scale of teacher target weekly loads; default: 3.25.",
    )
    parser.add_argument(
        "--teacher-max-load",
        type=int,
        default=12,
        help="Maximum weekly lectures per teacher; default: 12.",
    )
    parser.add_argument(
        "--co-teaching-probability",
        type=float,
        default=0.07,
        help="Probability that a section has two teachers; default: 0.07.",
    )
    parser.add_argument(
        "--shared-teacher-probability",
        type=float,
        default=0.5,
        help="Probability that a section reuses the preceding section's staff; default: 0.5.",
    )
    parser.add_argument(
        "--max-daily-load",
        type=int,
        default=4,
        help="Institutional daily teacher limit in time blocks; default: 4.",
    )
    parser.add_argument(
        "--students", type=int, default=500, help="Students per instance; default: 500."
    )
    parser.add_argument(
        "--courses-per-level",
        type=int,
        nargs=6,
        default=[4, 6, 6, 6, 6, 2],
        help="Course counts for the six levels; default: 4 6 6 6 6 2.",
    )
    parser.add_argument(
        "--level-decay",
        type=float,
        default=0.85,
        help="Geometric base-level decay factor; default: 0.85.",
    )
    parser.add_argument(
        "--teacher-availability",
        type=float,
        help="Probability that a block is available; default: 0.66. At least one block per day is retained.",
    )
    parser.add_argument(
        "--student-load-mean",
        type=float,
        help="Normal mean before rounding/truncation to 0-9 courses; default: 3.12.",
    )
    parser.add_argument(
        "--student-load-sd",
        type=float,
        help="Normal SD before rounding/truncation; default: 1.6.",
    )
    parser.add_argument(
        "--section-sd",
        type=float,
        default=0.8,
        help="Section-count normal SD before truncation to 1-4; default: 0.8.",
    )
    parser.add_argument(
        "--weekly-lectures-mean",
        type=float,
        default=2.59,
        help="Weekly lecture-count normal mean; default: 2.59.",
    )
    parser.add_argument(
        "--weekly-lectures-sd",
        type=float,
        help="Weekly lecture-count normal SD; default: 0.7; range: 2-4.",
    )
    parser.add_argument(
        "--preassignment-probability",
        type=float,
        default=0.10,
        help="Probability of attempting whole-course preassignment; default: 0.10.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Write accepted instance data, preserving existing datasets.

    Returns
    -------
    int
        Zero when all requested groups finish, or one when a candidate cap is
        reached. Invalid arguments raise SystemExit with exit code two.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.count < 1 or args.seed < 0 or args.construction_retries < 0:
        parser.error("count must be positive; seed and construction-retries must be nonnegative.")
    if args.max_candidates is not None and args.max_candidates < args.count:
        parser.error("max-candidates must be at least count.")
    groups = [DatasetGroup(value) for value in args.groups]
    if len(set(groups)) != len(groups):
        parser.error("Each group may be requested only once.")
    planned = []
    for group in groups:
        section_mean, sampling_sd, group_index = GROUP_PARAMETERS[group]
        options = GenerationOptions(
            courses_per_level=tuple(args.courses_per_level),
            teacher_load_mean=args.teacher_load_mean,
            teacher_load_sd=args.teacher_load_sd,
            teacher_max_load=args.teacher_max_load,
            co_teaching_probability=args.co_teaching_probability,
            shared_teacher_probability=args.shared_teacher_probability,
            max_daily_load=args.max_daily_load,
            student_count=args.students,
            section_mean=section_mean,
            section_sd=args.section_sd,
            weekly_lectures_mean=args.weekly_lectures_mean,
            weekly_lectures_sd=args.weekly_lectures_sd if args.weekly_lectures_sd is not None else 0.7,
            curricular_sampling_sd=sampling_sd,
            level_decay=args.level_decay,
            availability_probability=args.teacher_availability
            if args.teacher_availability is not None
            else 0.66,
            preassignment_probability=args.preassignment_probability,
            student_load_mean=args.student_load_mean if args.student_load_mean is not None else 3.12,
            student_load_sd=args.student_load_sd if args.student_load_sd is not None else 1.6,
        )
        try:
            options.validate()
        except ValueError as error:
            parser.error(str(error))
        directory = args.output / str(group)
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            parser.error(f"Destination must be absent or empty: {directory}")
        planned.append((group, group_index, options, directory))
    try:
        for group, group_index, options, directory in planned:
            directory.mkdir(parents=True, exist_ok=True)
            result = _generate_group(
                options, group=str(group), group_index=group_index,
                count=args.count, seed=args.seed, retries=args.construction_retries,
                max_candidates=args.max_candidates,
            )
            if result is None:
                return 1
            for number, (candidate_seed, data) in enumerate(result, start=1):
                destination = directory / f"instance_{number:03d}.json"
                with destination.open("x", encoding="utf-8") as stream:
                    json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
                    stream.write("\n")
                print(f"Saved {group}/{destination.name} (candidate seed {candidate_seed}).", flush=True)
    except OSError as error:
        parser.error(str(error))
    return 0

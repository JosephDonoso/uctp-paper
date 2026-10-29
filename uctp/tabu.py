"""Relocation Tabu Search with FIFO memory and strict-best aspiration."""

import random
import time
from dataclasses import dataclass
from math import ceil, sqrt

from .constraints import InstitutionalConstraints
from .domain import Instance, Solution
from .objective import SearchObjective
from .sectioning import StudentSectioning

type Move = tuple[int, int, int, int, int]


@dataclass
class _Candidate:
    solution: Solution
    move: Move | None
    objective: float


class TabuSearch:
    """Optimize X, optionally resolving the SSP for each feasible candidate.

    Notes
    -----
    The FIFO memory persists between calls, while the incumbent and stopping
    counters restart each call. Limits are checked after accepted moves. Ten
    consecutive empty/admissibility-failed neighborhoods terminate a call.
    """

    def __init__(
        self,
        instance: Instance,
        constraints: InstitutionalConstraints,
        objective: SearchObjective,
        sectioning: StudentSectioning,
        *,
        embedded: bool,
        neighborhood_size: int,
        max_stagnant_iterations: int,
        time_limit_seconds: float | None,
        collect_trace: bool = False,
    ) -> None:
        self.instance = instance
        self.constraints = constraints
        self.objective = objective
        self.sectioning = sectioning
        self.embedded = embedded
        self.neighborhood_size = neighborhood_size
        self.max_stagnant_iterations = max_stagnant_iterations
        self.time_limit_seconds = time_limit_seconds
        movable_count = sum(
            1
            for course in instance.courses
            for section in course.sections
            for lecture in section.lectures
            if lecture.preassigned_block is None
        )
        self._tabu_capacity = ceil(sqrt(movable_count))
        self._tabu_list: list[tuple[int, int, int, int]] = []
        self.completed_iterations = 0
        self.search_calls = 0
        self.collect_trace = collect_trace
        self.trace: list[dict] = []

    def solve(self, solution: Solution, *, max_iterations: int) -> str:
        """Replace X and Y with the best evaluated pair; return a stop reason."""
        self.search_calls += 1
        current = _Candidate(solution.copy(), None, self.objective.evaluate(solution))
        best = _Candidate(current.solution.copy(), current.move, current.objective)
        iteration_count = 0
        stagnant_count = 0
        failed_count = 0
        start_time = time.time()
        while True:
            neighborhood = self._generate_neighborhood(current, solution)
            selected = None
            while neighborhood:
                neighborhood.sort(key=lambda candidate: candidate.objective)
                candidate = neighborhood[0]
                if not self._is_tabu(candidate) or candidate.objective < best.objective:
                    selected = candidate
                    break
                neighborhood.remove(candidate)
            if selected is None:
                failed_count += 1
                if failed_count >= 10:
                    reason = "No admissible relocation found."
                    break
                continue
            failed_count = 0
            self._remember_move(selected)
            current = selected
            solution.X = selected.solution.X
            if current.objective < best.objective:
                best = _Candidate(current.solution.copy(), current.move, current.objective)
                stagnant_count = 0
            else:
                stagnant_count += 1
            iteration_count += 1
            self.completed_iterations += 1
            if self.collect_trace:
                self.trace.append(
                    {
                        "iteration": self.completed_iterations,
                        "search_block": self.search_calls,
                        "block_iteration": iteration_count,
                        "current_F": current.objective,
                        "block_best_F": best.objective,
                        "block_elapsed_seconds": time.time() - start_time,
                    }
                )
            if iteration_count >= max_iterations:
                reason = "Iteration limit reached."
                break
            if stagnant_count >= self.max_stagnant_iterations:
                reason = "Stagnation limit reached."
                break
            if (
                self.time_limit_seconds is not None
                and time.time() - start_time >= self.time_limit_seconds
            ):
                reason = "Time limit reached."
                break
        solution.X = best.solution.X
        solution.Y = best.solution.Y
        return reason

    def _generate_neighborhood(self, current: _Candidate, working: Solution) -> list[_Candidate]:
        neighborhood = []
        moves = [
            (*key, block)
            for key, active in current.solution.X.items()
            if active == 1
            and self.instance.courses[key[0]].sections[key[1]].lectures[key[2]].preassigned_block
            is None
            for block in range(len(self.instance.block_ids))
            if block != key[3]
        ]
        if not moves:
            return neighborhood
        for move in random.sample(moves, min(self.neighborhood_size, len(moves))):
            next_X = current.solution.X.copy()
            course_index, section_index, lecture_index, old_block, new_block = move
            next_X[course_index, section_index, lecture_index, old_block] = 0
            next_X[course_index, section_index, lecture_index, new_block] = 1
            working.X = next_X
            if self.constraints.is_feasible(working.X):
                if self.embedded:
                    objective = self.sectioning.solve(working)
                else:
                    objective = self.objective.evaluate(working)
                neighborhood.append(_Candidate(Solution(next_X, working.Y.copy()), move, objective))
        working.X = current.solution.X
        working.Y = current.solution.Y
        return neighborhood

    def _is_tabu(self, candidate: _Candidate) -> bool:
        if candidate.move is None:
            return False
        course_index, section_index, lecture_index, _, new_block = candidate.move
        for tab_course, tab_section, tab_lecture, forbidden_block in self._tabu_list:
            if (tab_course, tab_section, tab_lecture) == (
                course_index,
                section_index,
                lecture_index,
            ):
                if new_block == forbidden_block:
                    return True
        return False

    def _remember_move(self, candidate: _Candidate) -> None:
        if self._tabu_capacity <= 0 or candidate.move is None:
            return
        course_index, section_index, lecture_index, old_block, _ = candidate.move
        restriction = course_index, section_index, lecture_index, old_block
        if restriction not in self._tabu_list:
            if len(self._tabu_list) >= self._tabu_capacity:
                self._tabu_list.pop(0)
            self._tabu_list.append(restriction)

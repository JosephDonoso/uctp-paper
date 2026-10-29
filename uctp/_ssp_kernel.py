"""Numba kernel for the SSP; reproduces the pure-Python solver exactly.

The kernel runs the same algorithm as ``StudentSectioning`` in pure Python:
the same student order, the same course and branch ordering, the same
branch-and-bound with equality pruning, and the same local-search moves and
acceptance rule. Random draws are reproduced bit for bit by running CPython's
Mersenne Twister (``random.shuffle`` and ``_randbelow_with_getrandbits``) on
the state taken from ``random.getstate()``; the advanced state is written back
with ``random.setstate()``, so later draws are also unchanged.
"""

import numpy as np
from numba import njit

_N = 624
_M = 397
_MATRIX_A = 0x9908B0DF
_UPPER = 0x80000000
_LOWER = 0x7FFFFFFF
_MASK32 = 0xFFFFFFFF
_BIG = 1 << 60


# ---------------------------------------------------------------- CPython MT19937
@njit(cache=True)
def _genrand_uint32(mt, pos):
    if pos[0] >= _N:
        for kk in range(_N - _M):
            y = (mt[kk] & _UPPER) | (mt[kk + 1] & _LOWER)
            mt[kk] = mt[kk + _M] ^ (y >> 1) ^ (_MATRIX_A if y & 1 else 0)
        for kk in range(_N - _M, _N - 1):
            y = (mt[kk] & _UPPER) | (mt[kk + 1] & _LOWER)
            mt[kk] = mt[kk + (_M - _N)] ^ (y >> 1) ^ (_MATRIX_A if y & 1 else 0)
        y = (mt[_N - 1] & _UPPER) | (mt[0] & _LOWER)
        mt[_N - 1] = mt[_M - 1] ^ (y >> 1) ^ (_MATRIX_A if y & 1 else 0)
        pos[0] = 0
    y = mt[pos[0]]
    pos[0] += 1
    y ^= y >> 11
    y ^= (y << 7) & 0x9D2C5680
    y ^= (y << 15) & 0xEFC60000
    y ^= y >> 18
    return y & _MASK32


@njit(cache=True)
def _bit_length(n):
    k = 0
    while n:
        n >>= 1
        k += 1
    return k


@njit(cache=True)
def _randbelow(n, mt, pos):
    # CPython _randbelow_with_getrandbits; getrandbits(k <= 32) = word >> (32 - k).
    k = _bit_length(n)
    r = _genrand_uint32(mt, pos) >> (32 - k)
    while r >= n:
        r = _genrand_uint32(mt, pos) >> (32 - k)
    return r


@njit(cache=True)
def _shuffle(x, length, mt, pos):
    for i in range(length - 1, 0, -1):
        j = _randbelow(i + 1, mt, pos)
        tmp = x[i]
        x[i] = x[j]
        x[j] = tmp


# ------------------------------------------------------------------ student cost
@njit(cache=True)
def _cost(counts, n_blocks, tolerance):
    """Cost(W_s) over the multiset given by per-block counts (sorted order)."""
    inconvenience = 0
    any_block = False
    previous_day = -1
    previous_slot = 0
    first_slot = 0
    for block in range(n_blocks):
        for _ in range(counts[block]):
            any_block = True
            day = block // 7
            slot = block - 7 * day
            if day != previous_day:
                if previous_day >= 0 and first_slot <= 3 and previous_slot >= 4:
                    inconvenience += 1
                previous_day = day
                first_slot = slot
            else:
                gap = slot - previous_slot - 1
                inconvenience += 7 if gap < 0 else gap
            previous_slot = slot
    if not any_block:
        return 0.0
    if first_slot <= 3 and previous_slot >= 4:
        inconvenience += 1
    if inconvenience > tolerance:
        excess = inconvenience - tolerance
        return tolerance + excess * excess
    return float(inconvenience)


@njit(cache=True)
def _popcount(x):
    count = 0
    while x:
        x &= x - 1
        count += 1
    return count


# ------------------------------------------------------------------------- SSP
@njit(cache=True)
def solve_ssp(
    mt,
    pos,
    n_students,
    stu_ptr,
    stu_courses,
    stu_sorted,
    n_sections,
    sec_off,
    sb_ptr,
    sb_blocks,
    sec_mask,
    sec_distinct,
    n_blocks,
    tolerance,
    local_search,
    assigned_out,
    order_out,
    enrollments,
    costs,
):
    """Fill assigned_out (per student-course slot), enrollments and costs."""
    for i in range(enrollments.shape[0]):
        enrollments[i] = 0
    for i in range(n_students):
        costs[i] = 0.0
        order_out[i] = i
    _shuffle(order_out, n_students, mt, pos)

    max_k = 0
    for s in range(n_students):
        k = stu_ptr[s + 1] - stu_ptr[s]
        if k > max_k:
            max_k = k
    max_sec = 1
    for c in range(n_sections.shape[0]):
        if n_sections[c] > max_sec:
            max_sec = n_sections[c]
    branch_order = np.empty((max_k, max_sec), dtype=np.int64)
    position = np.empty(max_k, dtype=np.int64)
    branch_mask = np.zeros(max_k + 1, dtype=np.int64)
    branch_conf = np.zeros(max_k + 1, dtype=np.int64)
    chosen = np.empty(max_k, dtype=np.int64)
    best = np.empty(max_k, dtype=np.int64)
    counts = np.zeros(n_blocks, dtype=np.int64)
    shuffled = np.empty(max(max_k, 1), dtype=np.int64)

    for idx in range(n_students):
        s = order_out[idx]
        start = stu_ptr[s]
        k = stu_ptr[s + 1] - start
        if k == 0:
            continue
        # Branch order per depth: sections stably sorted by current enrollment.
        # Enrollments do not change during one student's search.
        for d in range(k):
            c = stu_sorted[start + d]
            ns = n_sections[c]
            for p in range(ns):
                branch_order[d, p] = p
            for a in range(1, ns):
                value = branch_order[d, a]
                key = enrollments[sec_off[c] + value]
                b = a - 1
                while b >= 0 and enrollments[sec_off[c] + branch_order[d, b]] > key:
                    branch_order[d, b + 1] = branch_order[d, b]
                    b -= 1
                branch_order[d, b + 1] = value
        # Depth-first branch and bound, identical visiting order and pruning.
        minimum = _BIG
        depth = 0
        position[0] = 0
        branch_mask[0] = 0
        branch_conf[0] = 0
        while depth >= 0:
            if minimum == 0:
                break  # every remaining branch would be pruned
            c = stu_sorted[start + depth]
            if position[depth] >= n_sections[c]:
                depth -= 1
                if depth >= 0:
                    position[depth] += 1
                continue
            p = branch_order[depth, position[depth]]
            sid = sec_off[c] + p
            if sec_distinct[sid]:
                conflicts = _popcount(sec_mask[sid] & branch_mask[depth])
            else:
                conflicts = 0
                for t in range(sb_ptr[sid], sb_ptr[sid + 1]):
                    if (branch_mask[depth] >> sb_blocks[t]) & 1:
                        conflicts += 1
            total = branch_conf[depth] + conflicts
            if total >= minimum:
                position[depth] += 1
                continue
            chosen[depth] = p
            if depth + 1 == k:
                minimum = total
                for d in range(k):
                    best[d] = chosen[d]
                position[depth] += 1
                continue
            branch_mask[depth + 1] = branch_mask[depth] | sec_mask[sid]
            branch_conf[depth + 1] = total
            depth += 1
            position[depth] = 0
        # Assign the minimum-conflict combination.
        for b in range(n_blocks):
            counts[b] = 0
        for d in range(k):
            c = stu_sorted[start + d]
            p = best[d]
            # Store by the student's original course slot.
            for t in range(k):
                if stu_courses[start + t] == c:
                    assigned_out[start + t] = p
                    break
            sid = sec_off[c] + p
            enrollments[sid] += 1
            for t in range(sb_ptr[sid], sb_ptr[sid + 1]):
                counts[sb_blocks[t]] += 1
        if local_search:
            for t in range(k):
                shuffled[t] = t  # slots in the original course order
            _shuffle(shuffled, k, mt, pos)
            has_multi = False
            for t in range(k):
                if n_sections[stu_courses[start + t]] > 1:
                    has_multi = True
            if has_multi:
                current_cost = _cost(counts, n_blocks, tolerance)
                improved = True
                while improved:
                    improved = False
                    for u in range(k):
                        slot = shuffled[u]
                        c = stu_courses[start + slot]
                        ns = n_sections[c]
                        if ns <= 1:
                            continue
                        current = assigned_out[start + slot]
                        cur_sid = sec_off[c] + current
                        for nxt in range(ns):
                            if nxt == current:
                                continue
                            nxt_sid = sec_off[c] + nxt
                            for t in range(sb_ptr[cur_sid], sb_ptr[cur_sid + 1]):
                                counts[sb_blocks[t]] -= 1
                            clash = False
                            for t in range(sb_ptr[nxt_sid], sb_ptr[nxt_sid + 1]):
                                if counts[sb_blocks[t]] > 0:
                                    clash = True
                                    break
                            if not clash:
                                for t in range(sb_ptr[nxt_sid], sb_ptr[nxt_sid + 1]):
                                    counts[sb_blocks[t]] += 1
                                next_cost = _cost(counts, n_blocks, tolerance)
                                if next_cost < current_cost:
                                    assigned_out[start + slot] = nxt
                                    enrollments[cur_sid] -= 1
                                    enrollments[nxt_sid] += 1
                                    current_cost = next_cost
                                    improved = True
                                    break
                                for t in range(sb_ptr[nxt_sid], sb_ptr[nxt_sid + 1]):
                                    counts[sb_blocks[t]] -= 1
                            for t in range(sb_ptr[cur_sid], sb_ptr[cur_sid + 1]):
                                counts[sb_blocks[t]] += 1
        costs[s] = _cost(counts, n_blocks, tolerance)


@njit(cache=True)
def assignment_costs(
    n_students, y_student, y_sid, sb_ptr, sb_blocks, n_blocks, tolerance, enrollments, costs
):
    """Per-student Cost(W_s) and section enrollments for a fixed Y (sorted by student)."""
    for i in range(enrollments.shape[0]):
        enrollments[i] = 0
    for i in range(n_students):
        costs[i] = 0.0
    counts = np.zeros(n_blocks, dtype=np.int64)
    m = y_student.shape[0]
    i = 0
    while i < m:
        s = y_student[i]
        for b in range(n_blocks):
            counts[b] = 0
        while i < m and y_student[i] == s:
            sid = y_sid[i]
            enrollments[sid] += 1
            for t in range(sb_ptr[sid], sb_ptr[sid + 1]):
                counts[sb_blocks[t]] += 1
            i += 1
        costs[s] = _cost(counts, n_blocks, tolerance)


def section_arrays(instance, section_blocks):
    """Flatten per-section blocks (course, section order) into CSR arrays and masks."""
    sizes = [len(course.sections) for course in instance.courses]
    total = sum(sizes)
    sb_ptr = np.zeros(total + 1, dtype=np.int64)
    flat = []
    masks = np.zeros(total, dtype=np.int64)
    distinct = np.zeros(total, dtype=np.bool_)
    sid = 0
    for course_index, size in enumerate(sizes):
        for section_index in range(size):
            blocks = section_blocks[course_index, section_index]
            flat.extend(blocks)
            mask = 0
            for block in blocks:
                mask |= 1 << block
            masks[sid] = mask
            distinct[sid] = mask.bit_count() == len(blocks)
            sid += 1
            sb_ptr[sid] = len(flat)
    return sb_ptr, np.array(flat, dtype=np.int64), masks, distinct

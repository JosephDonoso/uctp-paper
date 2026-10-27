# Instance generation

The command accepts entity counts and sampling parameters and writes only
instance data. There are no alternative calibration profiles or exported
generator metadata. Use Python 3.12.13.

Section counts (1–4), weekly lecture counts (2–4), and student course counts
(0–9) use independent rounded, truncated normals. All sections of a course
have the same weekly lecture count. Means and SDs describe the normals
before rounding/truncation, rather than exact final sample moments.

Students draw a base level using geometric decay. Every student uses the
fixed curricular sampling SD assigned to its dataset group.
Courses are selected without duplicates. Zero-load students remain in the data.

Teacher availability is sampled separately for each weekly block with
probability 0.7 of being available. If no block is available on a day, one
block is chosen randomly and made available. Teachers are assigned per course using
coverage and target-guided allocation, with one or two teachers shared by all
its sections. Weekly teacher load is an outcome, not a target. For each group, M is the largest weekly
teacher load across all its accepted instances. Every teacher in that group
uses the same daily limit, `floor(M / (N_days - 2)) + 1`, with `N_days = 5`. The group is revalidated if
candidate replacements change M, and files are saved only after the complete
group passes construction feasibility. All generated sessions use the course
teachers; separate teaching-assistant staffing is not modeled. This rule does
not guarantee feasibility: complete construction is still required before saving.

Preassignment probability selects whole courses for a feasible placement
attempt. It does not guarantee an exact final percentage. The CLI then checks
complete institutional construction feasibility before saving. This does not
require zero student conflicts or guarantee success for every later seed.

Preserve the command and printed candidate seeds to reproduce generated data.
Use `--max-candidates` to bound generation effort.

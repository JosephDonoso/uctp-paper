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
probability 0.66 of being available. If no block is available on a day, one
block is chosen randomly and made available.

Teachers are assigned per section. Each section receives one teacher, or two
with probability 0.07, and every assigned teacher teaches all lectures of the
section. Teachers are created with target weekly loads drawn from a rounded
normal with location 2 and scale 3.25, truncated to 2–12 lectures, until the
targets cover the weekly teaching demand; the number of teachers is therefore
an outcome of the demand. Courses are visited in random order. A later section
of a course reuses the staff of the preceding section with probability 0.5
when that staff stays within 12 weekly lectures; otherwise each place goes to
the teacher with the largest remaining target that stays within 12 lectures
(ties broken at random); if no existing teacher can take the section, a new
teacher is created with a target from the same distribution. No teacher exceeds
12 weekly lectures. Every teacher
uses the institutional daily limit of 4 blocks (`--max-daily-load`), and the
maximum number of teaching days follows from the weekly lecture count (3 days
for at most six lectures, 4 otherwise). All generated sessions use the section
teachers; separate teaching-assistant staffing is not modeled.

Preassignment probability selects whole courses for a placement attempt that
respects teacher availability, teacher overlaps, the daily limit and the
maximum number of teaching days. It does not guarantee an exact final percentage. The CLI then checks
complete institutional construction feasibility before saving. This does not
require zero student conflicts or guarantee success for every later seed.

Preserve the command and printed candidate seeds to reproduce generated data.
Use `--max-candidates` to bound generation effort.

## Calibration to the institutional instance

The default parameters calibrate BASE to the institutional instance
(`datasets/Real/real.json`) by matching expected values: the student-load mean
(3.12) gives 3.169 requested courses per student; the section-count location
(1.17, SD 0.8) gives 1.486 sections per course; the weekly-lecture mean (2.59)
gives 2.683 lectures per section; the availability probability (0.66) gives
34.0 % unavailable teacher blocks; and the curricular sampling SD (0.31)
reproduces the mean within-student SD of requested curricular levels (0.187;
institutional levels are curricular years within the student's program,
`tools/real_curricular_levels.json`). The teacher-load parameters reproduce the
institutional weekly blocks per teacher. Offer-Low and Offer-High use section
locations 0.36 and 1.64, which give 20 % fewer and more expected sections per
course than BASE; Dispersion-Low and Dispersion-High use half and 1.5 times the
BASE curricular SD (0.155 and 0.465). `tools/input_descriptives.py` reports the
resulting comparison.

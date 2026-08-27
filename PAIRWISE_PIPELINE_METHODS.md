# Pairwise conduction-velocity / wave-coherence pipeline — detailed methods

This document explains, step by step, exactly what `body_wave_analysis.py` and
its callers compute — the inclusion rules, the search/fit procedures, and the
equations behind every derived quantity (conduction velocity, R², direction
consistency, etc.). It reflects the code as of **2026-08-27**. Several
constants here (window size, R² threshold) have already been changed more
than once during development — where that's true, this doc says so
explicitly and points to the live constant in the source file rather than
just quoting a number, so it can't silently go stale.

This is a *supplement* to the frequency/amplitude pipeline documented in
`CLAUDE.md` (`worm_gait_analysis.py`), not a replacement — the two share the
same input data and the same forward-locomotion frame gate (Step 0 below),
but ask a different question. The frequency pipeline asks "is the neck bend
reaching the tail at all, and how often does it cycle." This pipeline asks
"is there a single, constant crawling-wave velocity that explains the timing
relationship across *all 13 segments at once*, not just neck vs. tail."

---

## 0. Input data

One `fullDataTable.csv` per recording (tab-separated, no header row; column
layout is `worm_gait_analysis.HEADER_LIST`). Relevant columns for this
pipeline:

- `curvature1` … `curvature13` — local bend angle at each of the 13 interior
  joints between the tracker's 14 body points (`curvature1` = joint nearest
  the head, `curvature13` = joint nearest the tail). Units assumed to be
  degrees (see `worm_gait_analysis.py`'s docstring for the caveat on this).
- `bodyAxisSpeed10` — signed, instantaneous body-axis-relative speed.
  Positive = forward locomotion, negative = reversal (confirmed with the
  data owner). Integer-valued on this tracker, observed range ~0–16.
- `omegaTurn`, `autofocusing`, `laserOn` — binary flags for sharp turns and
  imaging artifacts.

Frame rate: `FPS = 20` (frames/second), fixed for this tracker.

---

## Step 0 — Which frames get analyzed at all: the forward-locomotion mask

`worm_gait_analysis.forward_mask(df)` — a frame is **included** only if
**all** of the following hold:

1. `bodyAxisSpeed10 > 0` (moving forward, not reversing or stationary)
2. `omegaTurn == 0` (not a sharp/omega turn)
3. `autofocusing == 0` (not a camera-autofocus artifact frame)
4. `laserOn == 0`, if that column exists (not an optogenetic-stimulation frame)
5. `curvature2`, `curvature13`, and `bodyAxisSpeed10` are all non-`NaN`

A **bout** is a maximal contiguous run of frames passing this mask
(`worm_gait_analysis.contiguous_runs`). Everything downstream — filtering,
lag search, windowing — operates *only* within these bouts. Reverse
locomotion, turns, imaging artifacts, and untracked frames are never
analyzed by this pipeline; they show up as literal gaps (no row exists for
that time span at all, not a missing/NaN value in an existing row).

---

## Step 1 — Low-pass filtering (`body_wave_analysis.low_pass`)

Each of the 13 `curvature` columns is filtered **independently**:

- **Filter**: 2nd-order Butterworth low-pass, cutoff `LOWPASS_CUTOFF_HZ = 3.0`
  Hz, applied zero-phase (`scipy.signal.filtfilt`) so it doesn't introduce
  its own lag (which would corrupt the lag estimates in Step 2).
  Normalized cutoff: `Wn = cutoff / (FPS/2) = 3.0 / 10.0 = 0.3`.
- **Deliberately pure low-pass, no high-pass component** — slow postural
  drift is kept in, per the data owner's explicit instruction ("low
  frequency is the real movement"). This is a different filter than the
  0.05–3.0 Hz *band*-pass used in the separate frequency pipeline.
- **NaN handling**: `filtfilt` is an IIR (infinite-memory) filter — a single
  `NaN` anywhere in the input silently turns the *entire* output to `NaN`.
  To avoid this, each maximal contiguous run of non-`NaN` samples in a
  column is filtered separately; `NaN` gaps between runs stay `NaN` in the
  output. A run shorter than `filtfilt`'s minimum pad length
  (`3 * max(len(a), len(b)) + 1` filter taps) is passed through unfiltered
  (raw) rather than filtered, to avoid edge artifacts on data too short to
  filter safely. (Confirmed separately that no forward-locomotion bout ever
  straddles a `NaN` in any of the 13 curvature columns, so this never
  truncates a real bout — it only skips the gaps *between* bouts.)

---

## Step 2 — Adjacent-segment lag/correlation search

This is the core measurement step, and the one most recently debugged (see
§9, "Known issues already fixed").

### 2a. Estimating a local reference period

For each analysis window (see Step 6 for what "window" means),
`estimate_dominant_period()` computes the dominant oscillation period from
the **middle segment** (segment 7 of 13), as a representative reference for
the whole window:

```
freqs, psd = Welch_PSD(curvature7[window], fs=FPS, nperseg=min(window_length, FPS*10))
period = 1 / freqs[argmax(psd within 0.05-3.0 Hz)]
```

Returns `None` if the window is shorter than 2 seconds or the PSD has no
positive power in that band (in which case a fixed fallback is used below).

### 2b. Setting the search range for each adjacent pair

```
max_lag_sec = clip(period * ADJACENT_LAG_FRACTION, ADJACENT_LAG_MIN_SEC, ADJACENT_LAG_MAX_SEC)
            = clip(period * 0.4,                    0.5,                  3.0)          seconds
```

(`max_lag_sec = ADJACENT_LAG_MAX_SEC = 3.0s` if the period estimate failed.)
This range is deliberately **narrow** — capped to a fraction of the local
period — so the search cannot "jump" to the wrong oscillation cycle. An
earlier version searched a wide, fixed ±20s range per pair (mirroring the
neck/tail-only propagation check) and it catastrophically failed: with 78
independent wide searches, many locked onto the wrong cycle, and
wave-coherence R² sat near 0 almost everywhere despite strong pairwise
correlations. Restricting to *adjacent* segments only, with this
period-scaled range, was the fix.

### 2c. The per-pair search itself (`narrow_lag_corr`)

For each of the 12 **adjacent** pairs (segment *i*, segment *i+1*, for
*i* = 1…12 — never a non-adjacent pair; see Step 3 for how those are
handled), search over integer lags (in frames) and keep whichever lag
maximizes `|correlation|`:

```
for lag in [min_lag_frames .. max_lag_frames]:
    a, b = curvature_i, curvature_(i+1), shifted against each other by `lag` frames
    corr(lag) = Pearson_correlation(a, b)     # numpy.corrcoef
best_lag  = argmax_lag |corr(lag)|
best_corr = corr(best_lag)
```

**Sign convention**: for `lag > 0`, `curvature_(i+1)` is shifted *later* —
i.e. positive lag means segment *i+1* (posterior) is delayed relative to
segment *i* (anterior). This is the expected sign for an anterior-to-
posterior wave during forward locomotion.

**Search range is restricted to non-negative lags only**
(`min_lag_sec = 0`, i.e. `lag ∈ [0, max_lag_frames]`, not the full
`[-max_lag_frames, +max_lag_frames]`). This is the fix made 2026-08-27 (see
§9) — it's safe specifically *because* every window analyzed here already
comes from a confirmed forward-locomotion bout (Step 0), so the propagation
direction is already known independently of this correlation search; it
isn't an assumption baked in to get a desired answer.

A minimum of `max(4, FPS//2) = 10` overlapping samples is required for a
lag to be considered at all.

---

## Step 3 — Building the full 13×13 lag/correlation matrices

`pairwise_matrix_for_window()` — lags for the 78 possible segment pairs
(13 choose 2) are **never** searched directly. Only the 12 adjacent pairs are
measured (Step 2); every other pair's lag is built by **summing** the
adjacent lags between them:

```
cum[0] = 0
cum[k] = adjacent_lag_sec[0] + adjacent_lag_sec[1] + ... + adjacent_lag_sec[k-1]   for k = 1..12

lag_mat[i, j] = cum[j] - cum[i]      # signed lag of segment j relative to segment i
              # positive => j (more posterior) is delayed relative to i
```

`corr_mat[i, j]` for a **non-adjacent** pair is *not* a measured correlation
— it's the **minimum `|corr|` among the adjacent links on the path from i to
j** (the confidence of the weakest link the reconstructed lag depends on).
This is what "strong pair" filtering (Step 4) actually uses.

---

## Step 4 — Per-window summary statistics (`summarize_window`)

Restricted to the 78 upper-triangle pairs (*i* < *j*) so distance and lag
share one consistent direction convention.

- **"Strong" pairs**: `|corr_mat[i,j]| >= MIN_PROPAGATION_CORR (0.3)`.
  Weak-link pairs are excluded from the fit — they'd just inherit scatter
  from a noisy segment-to-segment measurement.
- **Gate**: if fewer than **5** strong pairs exist, no fit is attempted at
  all (see Step 5 for what happens to R²/velocity in that case).

Otherwise, a **linear least-squares fit** of lag vs. distance across the
strong pairs only:

```
distance_ij = j - i                       # in segment-index units, 1..12
lag_ij ≈ slope * distance_ij + intercept   # numpy.polyfit, degree 1
```

### Conduction velocity

```
conduction_velocity_segments_per_s_raw = 1 / slope     # slope units: seconds/segment
```

Slope has units of **seconds of delay per segment of distance**, so its
reciprocal is segments of body length traveled per second — the speed at
which the bending wave travels along the body. (Not a real-world speed in
mm/s — the tracker doesn't give a segment-length calibration here.)

### R² ("wave coherence")

```
predicted_ij = slope * distance_ij + intercept
SS_res = Σ (lag_ij - predicted_ij)²                      over strong pairs
SS_tot = Σ (lag_ij - mean(lag over strong pairs))²       over strong pairs
R²     = 1 - SS_res / SS_tot          (if SS_tot > 0, else NaN — see §9)
```

R² measures how well a **single constant velocity** explains the timing
relationship across the whole body at once — it's the number to trust;
velocity is only meaningful conditional on R² being high.

### Direction consistency

```
majority_sign = sign(slope)
direction_consistency = fraction of strong pairs where sign(lag_ij) == majority_sign
                       (NaN if slope == 0)
```

Fraction of pairs whose lag sign agrees with the fitted (majority)
direction — a sanity check independent of the R² fit quality.

---

## Step 5 — Reliability gates

Three independent gates, all of which must pass before a velocity number is
ever reported. Each was added in response to a specific failure mode found
empirically (see the design-note comments at each constant in
`body_wave_analysis.py` for the exact numbers behind each decision):

| Gate | Threshold | Why |
|---|---|---|
| **R² floor** | `R² >= MIN_R2_FOR_VELOCITY` (pipeline default **0.5**; the current batch run in `run_batch_pairwise_analysis.py` uses **0.7** — see §8) | velocity = 1/slope blows up near a zero slope; a bad fit can't be trusted to invert. |
| **Window duration floor** | `window_duration_s >= MIN_WINDOW_SEC_FOR_FIT` (**3.0s**, = `worm_gait_analysis.MIN_BOUT_SEC`) | A short window can hit a high R² by chance (a real example: a 0.6s/12-frame window scored R²=0.71 with a nonphysical ~19 segments/s). |
| **Plausibility ceiling** | `\|velocity\| <= MAX_PLAUSIBLE_VELOCITY_SEGMENTS_PER_S` (**5.0** segments/s) | Even the two gates above didn't catch everything at full-batch scale — a backstop, not a substitute for the other two. |

```
velocity_reliable = (R² is not NaN) AND (R² >= min_r2_for_velocity)
                     AND (velocity_raw is not NaN) AND (|velocity_raw| <= 5.0)

conduction_velocity_segments_per_s = velocity_raw  if velocity_reliable else NaN
```

If the window fails the duration floor, `r_squared`, `velocity`, and
`direction_consistency` are all force-reset to `NaN`/`False` regardless of
what the fit above computed — a short window's fit isn't trusted at all,
even if it happened to look good.

### The `r_squared_source` field — what a "0" or a missing value actually means

After the gates above, `r_squared` can be `NaN` for two structurally
different reasons: too few strong pairs, or too short a window. Rather than
leaving both as an undifferentiated `NaN`, each window is tagged with **why**:

```
if r_squared is NaN:
    if |mean(bodyAxisSpeed10 over the window)| <= NEAR_ZERO_SPEED_THRESHOLD (1):
        r_squared = 0.0
        r_squared_source = "near_zero_speed_default"
    else:
        r_squared_source = "unavailable"     # r_squared stays NaN
else:
    r_squared_source = "fit"
```

- **`"fit"`** — a real pairwise fit was computed (may still be low, or fail
  `velocity_reliable`).
- **`"near_zero_speed_default"`** — no fit was possible, but the worm was
  essentially not translating during this window (mean `bodyAxisSpeed10`
  magnitude ≤ 1, the smallest non-zero value on this tracker's scale, even
  though every frame individually passed `bodyAxisSpeed10 > 0`). "No wave
  coherence could be established" has a physical explanation here (no real
  movement to correlate), so `0.0` is a meaningful value, not a guess.
- **`"unavailable"`** — no fit was possible **despite** real movement.
  `r_squared` stays `NaN` — the true value is genuinely unknown, not zero.

`conduction_velocity_*` is **never** defaulted this way — there's no slope
to invert in a "no fit" case, and defaulting it to 0 would misrepresent "no
measurement" as "zero velocity."

---

## Step 6 — Sliding-window construction (`sliding_windows_for_bout`)

Within each forward-locomotion bout, windows are `win_sec` long, spaced
`step_sec` apart, with one exception:

```
if bout_duration <= win_sec:
    windows = [ (entire bout) ]        # one window, SHORTER than win_sec
else:
    windows = [ (bs+0, bs+win), (bs+step, bs+step+win), ... ]   # regular win_sec-long grid
    if the last regular window doesn't reach the bout's end:
        windows.append( (bout_end - win, bout_end) )            # still win_sec long, just realigned
```

This is why window durations vary in the output: a forward bout shorter than
the target window (e.g. the worm sustains forward movement for only 2s
before turning) still gets analyzed as one window spanning whatever duration
is actually available, rather than being skipped or padded with data that
doesn't exist. This is also precisely why the duration floor in Step 5
exists.

**`win_sec`/`step_sec` are NOT one single global constant** — they differ by
caller:

| Caller | win_sec / step_sec | Source |
|---|---|---|
| `analyze_file_pairwise()` default | 10s / 5s | `worm_gait_analysis.PROP_WINDOW_SEC` / `PROP_STEP_SEC` — shared with the unrelated neck/tail propagation heuristic; changing these globally would silently change that pipeline too. |
| `run_batch_pairwise_analysis.py` (the actual batch CSVs) | **check `PAIRWISE_WINDOW_SEC`/`PAIRWISE_STEP_SEC` in the file** — has been changed several times during development (5s→8s→5s as of this writing) | Set at the call site specifically so it doesn't affect the shared 10s/5s default above. |
| `plot_full_video_wave_summary.py`, `plot_pairwise_check.py` | 10s/5s default, overridable via `--win-sec`/`--step-sec` CLI flags | Independent of the batch script — pass matching flags to cross-check a diagnostic plot against a specific batch run. |

`MIN_WINDOW_SEC_FOR_FIT` (3.0s) does **not** scale with `win_sec` — if
`win_sec` is set below 3s, every window fails the duration floor (falling
back to the `r_squared_source` logic above).

---

## Step 7 — Batch aggregation (`run_batch_pairwise_analysis.py`)

Runs `analyze_file_pairwise()` over every `fullDataTable.csv` under
`fullDataTables_diffStagingMethods/0coverslip_spacers/` (52 files as of this
writing: N2 and PHX9753 genotypes), producing three CSVs.

**`gait_pairwise_windows.csv`** — one row per sliding window, every file
(the columns listed in `WINDOW_FIELDS`, which includes every field described
above except the matrices themselves).

**`gait_pairwise_by_file.csv`** — one row per file:

```
n_windows                        = count of all windows for this file
n_windows_long_enough            = count where window_long_enough == True
n_windows_reliable_velocity      = count where velocity_reliable == True
frac_windows_reliable_velocity   = n_windows_reliable_velocity / n_windows
mean_r_squared, median_r_squared = mean/median of r_squared across ALL windows
                                    (pandas skips NaN => "unavailable" windows excluded;
                                     "near_zero_speed_default" 0.0's ARE included)
mean/median/std_velocity_segments_per_s = across velocity_reliable windows only
mean_direction_consistency       = mean across windows with a real fit (NaN skipped)
```

**`gait_pairwise_by_group.csv`** — one row per `staging_method × genotype`,
aggregating the **per-file** statistics above (a mean of file-level means,
not re-derived from pooled raw windows — so every file counts equally
regardless of how many windows it contributed).

The script is **resumable**: if `gait_pairwise_windows.csv` already exists,
any filename already present in it is skipped on the next run. This check
is purely by filename — it does **not** detect that `PAIRWISE_WINDOW_SEC` or
any other setting changed since the file was written. **Whenever a setting
that changes what gets computed per-file is changed, the three CSVs must be
deleted (or moved aside) before re-running**, or the run will silently do
nothing.

---

## Step 8 — Over-time aggregation across videos (`plot_over_time_by_genotype.py`)

Three figures, all built from `gait_pairwise_windows.csv`:

1. **`plot_coherence_overlay`** — one panel per genotype: faint individual
   per-window R² points (not connected across a video's own gaps) plus a
   bold across-video mean trend line.
2. **`plot_coherence_average`** — same trend lines as (1), no individual
   points, both genotypes overlaid on one axes.
3. **`plot_velocity_average`** — same style as (2), but for conduction
   velocity, restricted to `velocity_reliable == True` windows only.

**Binning**: `time_bin = floor(window_mid_s / BIN_SEC) * BIN_SEC`, where
`window_mid_s` is the window's midpoint and `BIN_SEC` (currently **30**
seconds — check the live constant) is independent of the underlying
sliding-window size used to produce the CSV.

**"Mean of video means" (two-stage averaging)**, used for every trend line:

```
Stage 1: for each (video, time_bin), take the mean of that video's own windows in that bin
Stage 2: for each time_bin, mean = mean of the Stage-1 per-video means (across videos)
         SEM = std(per-video means, ddof=1) / sqrt(n_videos)     [NaN if only 1 video]
```

This means a video that happens to contribute many windows to a bin doesn't
outweigh one that contributes few — every video counts equally.

**Reliability of a bin itself**: a bin's trend-line segment is drawn solid
only if `n_videos_contributing >= MIN_VIDEOS_PER_BIN` (**3**); otherwise
dotted, since recordings vary a lot in length (190–930s in this dataset) and
the number of contributing videos shrinks at later times — a dotted segment
from 2 videos isn't the same kind of signal as a solid one from 26.

---

## 9. Known issues already found and fixed (chronological)

1. **Wide, per-pair signed lag search (original design)** — searching every
   one of the 78 pairs over a wide ±20s range let many latch onto the wrong
   oscillation cycle. **Fixed**: adjacent-only search, range capped to a
   fraction of the local period (Step 2b/2c), non-adjacent lags built by
   summing (Step 3).
2. **Sign-flipped adjacent correlations (found 2026-08-27)** — even within
   the narrow, period-capped range, `narrow_lag_corr` picked whichever lag
   maximized `|correlation|` regardless of sign. Because curvature is an
   oscillating signal, a handful of adjacent pairs could spuriously lock
   onto a strong *anti-phase* match (e.g. lag=−2.7s, corr=−0.98) instead of
   the true, similarly-strong in-phase match at a positive lag. Since lags
   are reconstructed by **cumulatively summing** adjacent lags, even 2–3
   sign-flipped links (out of 12) wrecked an otherwise clean,
   strongly-propagating lag-vs-distance fit — this was the cause of R²
   values near 0 in windows whose kymograph showed unmistakably clean
   propagation. **Fixed**: restrict the adjacent-pair search to non-negative
   lags only (Step 2c), since the propagation direction is already known
   independently from the forward-locomotion mask. Dataset-wide validation:
   `mean_direction_consistency` rose from ~0.77 to ~0.98 after this fix.
3. **R² defaulting** — went through two iterations: first a blanket
   "default every unfittable window's R² to 0," then narrowed to only
   default when the window was confirmed near-stationary (Step 5's
   `r_squared_source` logic), leaving genuinely-unknown cases as `NaN`
   rather than a guessed `0`.

## 10. Open questions / not yet revisited

- `MIN_R2_FOR_VELOCITY` (0.5) and `MAX_PLAUSIBLE_VELOCITY_SEGMENTS_PER_S`
  (5.0) were calibrated against the **pre-fix** (sign-flip bug present)
  behavior. Now that R² runs much higher and more consistently after the
  fix above, these thresholds haven't been re-validated against the
  corrected algorithm.
- Whether to fold this pairwise wave-coherence measure into the frequency
  pipeline's neck/tail propagation gate (replace) or keep both as separate,
  complementary analyses (supplement) is still undecided.
- Segment 13 (tail) has broadband noise that doesn't fully separate
  spectrally from real movement even after the 3 Hz low-pass — flagged in
  `CLAUDE.md`, not addressed by anything in this pipeline.
- The `r_squared_source` breakdown (fit / near_zero_speed_default /
  unavailable) is only visible in the finest-grained `gait_pairwise_windows.csv`
  and in `plot_full_video_wave_summary.py`'s panel markers — it is **not**
  yet broken out separately in `gait_pairwise_by_file.csv` /
  `gait_pairwise_by_group.csv`, which pool "fit" and
  "near_zero_speed_default" windows together under one `r_squared` mean.

---

## 11. File-by-file map

| File | Role |
|---|---|
| `worm_gait_analysis.py` | Column headers, `FPS`, `forward_mask` (Step 0), `contiguous_runs` — shared foundation for both this pipeline and the separate frequency/amplitude pipeline. |
| `body_wave_analysis.py` | Steps 1–5. `analyze_file_pairwise(path, win_sec, step_sec, min_r2_for_velocity)` is the single entry point everything else calls. |
| `run_batch_pairwise_analysis.py` | Batch driver + Steps 6–7. Produces the three `gait_pairwise_*.csv` files. |
| `plot_pairwise_check.py` | Single-file diagnostic: R²/velocity over time, plus best- and worst-window 13×13 lag heatmaps + lag-vs-distance fit plots. |
| `plot_full_video_wave_summary.py` | Single-file diagnostic: full-recording kymograph + R² (with `r_squared_source` markers) + velocity, all time-aligned. |
| `plot_over_time_by_genotype.py` | Step 8: cross-video aggregate comparison between genotypes. |

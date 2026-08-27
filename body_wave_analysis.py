"""
body_wave_analysis.py

Step 1 (low-pass filter) and Step 2 (full pairwise cross-correlation) of the
whole-body conduction workflow described in workflow_plan_body_wave_analysis.md.

Decisions locked in by the data owner (2026-08-26):
  - Step 1 filter: FIXED cutoff (not PSD-derived).
  - Step 1 filter type: pure low-pass (no low-frequency floor) -- slow
    postural drift is kept in, per "low frequency is the real movement."

Everything else in this module follows the plan's recommended defaults:
  - zero-phase filtering (filtfilt), to avoid corrupting the lag estimates
    computed in step 2.
  - time-resolved (sliding-window) pairwise cross-correlation, reusing the
    same window/step sizes already validated for the neck/tail propagation
    fix in worm_gait_analysis.py (10s window, 5s step).
  - wide signed lag search per pair, same style as wga.bout_propagation().
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch

import worm_gait_analysis as wga

N_SEGMENTS = 13
CURVATURE_COLS = [f"curvature{i}" for i in range(1, N_SEGMENTS + 1)]

# --- Step 1: low-pass filter ---------------------------------------------
# Fixed cutoff, matching the existing bandpass's upper edge in
# worm_gait_analysis.py (BANDPASS_HIGH_HZ). Edit this single constant to
# change the cutoff everywhere in this module.
LOWPASS_CUTOFF_HZ = 3.0
LOWPASS_ORDER = 2


def low_pass(signal, fps=wga.FPS, cutoff=LOWPASS_CUTOFF_HZ, order=LOWPASS_ORDER):
    """Zero-phase Butterworth low-pass. Pure low-pass -- no high-pass floor,
    so slow postural drift passes through untouched, per the data owner's
    explicit choice to keep 'everything slow.'

    NaN-safe: `filtfilt` is an IIR (infinite-memory) filter, so a single NaN
    anywhere in the input silently turns the ENTIRE output to NaN, not just
    the missing frame -- this is not a hypothetical edge case, it happens on
    real files here (frames outside forward-locomotion bouts are ~5-10% NaN
    in these recordings). Fix: filter each maximal run of finite values
    independently and leave NaN gaps as NaN in the output. Confirmed
    separately that forward-locomotion bouts themselves never straddle a
    NaN in any of the 13 curvature columns, so this never silently truncates
    a bout -- it only skips over the NaN frames between bouts."""
    signal = np.asarray(signal, dtype=float)
    out = np.full_like(signal, np.nan)
    b, a = butter(order, cutoff / (fps / 2.0), btype="low")
    min_len = 3 * max(len(a), len(b)) + 1  # scipy filtfilt's default pad requirement
    finite = ~np.isnan(signal)
    for s, e in wga.contiguous_runs(finite):
        seg = signal[s:e]
        if len(seg) > min_len:
            out[s:e] = filtfilt(b, a, seg)
        else:
            out[s:e] = seg  # too short to filter without artifacts; pass through raw
    return out


def filter_all_segments(df, cols=CURVATURE_COLS, fps=wga.FPS):
    """Returns a DataFrame, same shape as df[cols], with each column
    low-pass filtered independently."""
    out = {c: low_pass(df[c].values, fps=fps) for c in cols}
    return pd.DataFrame(out, index=df.index)


# --- Step 2: full pairwise cross-correlation ------------------------------
#
# DESIGN NOTE (found via the diagnostic plots, fixed 2026-08-26):
# The first version of this searched every one of the 78 pairs over a wide,
# signed lag range (+-20s, same style as wga.bout_propagation()). That works
# for a single neck/tail pair -- you only care whether *a* propagating
# relationship exists, not which cycle it's measured on. It does NOT work
# pairwise: bending is periodic, so cross-correlation vs. lag has a peak
# roughly once per cycle, and with 78 independent wide searches many of them
# latch onto the wrong cycle (a real example: at a single distance, lag
# estimates scattered from -3.3s to +3.1s -- nearly the full period range,
# not clustered near a true value). The symptom was unmistakable once
# plotted: wave-coherence R^2 sat near 0 almost everywhere despite most
# pairs individually passing the |corr| strength threshold, and conduction
# velocity occasionally spiked to thousands of segments/s.
#
# Fix: never wide-search a non-adjacent pair. Only search ADJACENT segments
# (1-2, 2-3, ..., 12-13), with the lag search range capped to a fraction of
# the window's own dominant period -- narrow enough that it cannot jump to
# a neighboring cycle. Lags for every non-adjacent pair are then built by
# summing the adjacent lags between them, never by direct search. This also
# turns the lag-vs-distance R^2 into a more meaningful number: it now
# measures how *constant* the segment-to-segment delay is along the body,
# rather than being vulnerable to search-range artifacts.

ADJACENT_LAG_FRACTION = 0.4   # search range = this fraction of the local period
ADJACENT_LAG_MIN_SEC = 0.5    # floor, in case period estimation is noisy/short
ADJACENT_LAG_MAX_SEC = 3.0    # ceiling, in case period estimation fails high
PERIOD_ESTIMATE_LOW_HZ = 0.05
PERIOD_ESTIMATE_HIGH_HZ = 3.0


def estimate_dominant_period(signal, fps=wga.FPS, low=PERIOD_ESTIMATE_LOW_HZ, high=PERIOD_ESTIMATE_HIGH_HZ):
    """Welch-PSD peak frequency within [low, high] Hz, returned as a period
    in seconds. Returns None if the window is too short/flat to tell."""
    signal = np.asarray(signal, dtype=float)
    if len(signal) < fps * 2:
        return None
    freqs, psd = welch(signal, fs=fps, nperseg=min(len(signal), int(fps * 10)))
    band = (freqs >= low) & (freqs <= high)
    if not band.any() or not np.any(psd[band] > 0):
        return None
    f_peak = freqs[band][np.argmax(psd[band])]
    return (1.0 / f_peak) if f_peak > 0 else None


def narrow_lag_corr(sig_a, sig_b, fps=wga.FPS, max_lag_sec=ADJACENT_LAG_MAX_SEC, min_lag_sec=None):
    """Same signed-range/best-|corr| search as wga.bout_propagation(), just
    parameterized on a caller-supplied (narrow, for adjacent pairs) range
    instead of the wide MAX_PROPAGATION_LAG_SEC. Returns (best_lag_frames, best_corr).

    min_lag_sec defaults to -max_lag_sec (search the full signed range).
    DESIGN NOTE (data owner, 2026-08-26): adjacent_lags_for_window always
    calls this with min_lag_sec=0 instead, restricting the search to
    non-negative lags only. Found via a real example (a 10s window with
    R^2=0.002 despite every individual adjacent pair correlating at
    |corr|>=0.84 on the unfiltered kymograph): curvature is an oscillating
    signal, so "best |corr|, either sign" can lock onto a strong ANTI-phase
    match (e.g. lag=-2.7s, corr=-0.98) instead of the true, similarly-strong
    in-phase match at a positive lag (lag=+1.4s, corr=+0.97, confirmed by
    re-running the search restricted to lag>=0 on that exact pair) --
    checked, this happened on 3 of 12 adjacent pairs in that window alone.
    Because lags are reconstructed by CUMULATIVELY SUMMING adjacent lags,
    even a couple of sign-flipped links wreck an otherwise-clean, strongly
    propagating lag-vs-distance line. This is only safe to do because the
    caller already knows the propagation direction independently -- every
    window analyzed here comes from a forward-locomotion bout (positive
    bodyAxisSpeed10, confirmed with the data owner), so anterior-to-
    posterior (non-negative lag, per this module's own sign convention --
    see pairwise_matrix_for_window) is not a guess, it's already
    established before this function ever runs."""
    if min_lag_sec is None:
        min_lag_sec = -max_lag_sec
    max_lag = max(1, int(round(max_lag_sec * fps)))
    max_lag = min(max_lag, len(sig_a) // 3) if len(sig_a) >= 3 else 0
    min_lag = int(round(min_lag_sec * fps))
    min_lag = max(min_lag, -max_lag)
    best_lag, best_corr, best_abs_corr = 0, 0.0, -1.0
    for lag in range(min_lag, max_lag + 1):
        if lag < 0:
            a, b = sig_a[-lag:], sig_b[: len(sig_b) + lag]
        elif lag > 0:
            a, b = sig_a[: len(sig_a) - lag], sig_b[lag:]
        else:
            a, b = sig_a, sig_b
        if len(a) < max(4, fps // 2):
            continue
        c = np.corrcoef(a, b)[0, 1]
        if not np.isnan(c) and abs(c) > best_abs_corr:
            best_lag, best_corr, best_abs_corr = lag, c, abs(c)
    return best_lag, best_corr


def adjacent_lags_for_window(filtered_df, ws, we, fps=wga.FPS, cols=CURVATURE_COLS):
    """Lag (seconds) and corr for each of the n-1 adjacent-segment pairs in
    one window, using a search range capped to a fraction of that window's
    own dominant period (estimated from the middle segment, as a
    representative reference for the whole window).

    Search is restricted to non-negative lags (min_lag_sec=0) -- see
    narrow_lag_corr's design note: every caller of this function only ever
    analyzes forward-locomotion windows, where the propagation direction
    (anterior leads posterior) is already known independently, so this
    isn't assuming the answer -- it's using information the signed
    best-|corr| search doesn't otherwise have, and which was letting it
    lock onto strong-but-spurious anti-phase matches."""
    n = len(cols)
    ref = filtered_df[cols[n // 2]].values[ws:we]
    period = estimate_dominant_period(ref, fps=fps)
    if period is None:
        max_lag_sec = ADJACENT_LAG_MAX_SEC
    else:
        max_lag_sec = float(np.clip(period * ADJACENT_LAG_FRACTION, ADJACENT_LAG_MIN_SEC, ADJACENT_LAG_MAX_SEC))

    adj_lag_sec = np.full(n - 1, np.nan)
    adj_corr = np.full(n - 1, np.nan)
    for i in range(n - 1):
        a = filtered_df[cols[i]].values[ws:we]
        b = filtered_df[cols[i + 1]].values[ws:we]
        lag, corr = narrow_lag_corr(a, b, fps=fps, max_lag_sec=max_lag_sec, min_lag_sec=0.0)
        adj_lag_sec[i] = lag / fps
        adj_corr[i] = corr
    return adj_lag_sec, adj_corr, max_lag_sec


def pairwise_matrix_for_window(filtered_df, ws, we, fps=wga.FPS, cols=CURVATURE_COLS):
    """13x13 signed-lag (seconds) and correlation matrices for one window,
    built from adjacent-pair measurements only (see design note above) --
    NOT from a direct wide search on every pair.

    Sign convention: lag_mat[i, j] for i < j is the lag of segment j
    relative to segment i. Positive = j is delayed relative to i, i.e. the
    bend reaches the more posterior segment later -- the expected sign
    during anterior-to-posterior (forward-locomotion) wave propagation.
    lag_mat[j, i] = -lag_mat[i, j] (antisymmetric), diagonal is 0/1.

    corr_mat[i, j] for a non-adjacent pair is NOT a measured correlation --
    it's the minimum |corr| among the adjacent links on the path from i to
    j, i.e. the confidence of the weakest link the reconstructed lag
    depends on. This is what "strong pair" filtering uses downstream.
    """
    n = len(cols)
    adj_lag_sec, adj_corr, max_lag_sec = adjacent_lags_for_window(filtered_df, ws, we, fps=fps, cols=cols)

    cum = np.concatenate([[0.0], np.cumsum(adj_lag_sec)])  # cum[k] = lag of segment k relative to segment 0
    lag_mat = cum[np.newaxis, :] - cum[:, np.newaxis]      # lag_mat[i, j] = cum[j] - cum[i]

    # path-minimum |corr| between i and j, from the adjacent links only
    abs_adj = np.abs(adj_corr)
    corr_mat = np.full((n, n), np.nan)
    np.fill_diagonal(corr_mat, 1.0)
    for i in range(n):
        for j in range(i + 1, n):
            link_strengths = abs_adj[i:j]
            weakest = np.nanmin(link_strengths) if len(link_strengths) else np.nan
            corr_mat[i, j] = weakest
            corr_mat[j, i] = weakest

    return lag_mat, corr_mat, adj_lag_sec, adj_corr, max_lag_sec


# Minimum fit quality (R^2 of the lag-vs-distance fit) required before a
# velocity number is reported at all. DESIGN NOTE (found 2026-08-26, via
# user question about seeing velocities of +100/-300 seg/s): velocity is
# 1/slope, and when the true per-segment delay is near zero -- meaning
# there's no real linear lag-vs-distance relationship in that window, i.e.
# a bad fit -- the inversion blows up: tiny noise in the slope estimate
# produces enormous, sign-flippy "velocities" that are pure numerical
# artifact, not fast conduction. Checked directly on real data: in one
# example file, every window with |velocity| > 20 seg/s had R^2 <= 0.017,
# and every window with R^2 >= 0.5 had |velocity| <= 2.33 seg/s -- a clean
# separation. 0.5 sits well inside that gap. Below this threshold, velocity
# is reported as NaN (kept as conduction_velocity_segments_per_s_raw for
# debugging) rather than shown as if it were a real measurement.
MIN_R2_FOR_VELOCITY = 0.5

# Hard plausibility ceiling on |velocity|, applied IN ADDITION to the R^2 and
# window-duration gates above. DESIGN NOTE (found 2026-08-26, via the
# across-video "velocity over time" aggregate plot): the R^2/duration gates
# were validated on one file and don't fully generalize -- across the full
# 71-file, 4673-window batch, 2 windows still slipped through with R^2 just
# over 0.5 and durations just over the 3s floor, but velocities of 11.5 and
# 6.4 seg/s. Both were short (3.3-3.7s) windows where a marginal fit can
# still look "good enough" by chance. Checked directly: of 1334
# gate-passing windows, only these 2 exceed 5 seg/s; every legitimately
# validated window (10s duration, high R^2, checked by eye against
# kymographs) tops out around 2.6-3.3 seg/s. One bad window, averaged
# equally with several good ones in a per-video-then-across-video mean,
# was enough to visibly distort an aggregate plot -- hence a third,
# independent gate here rather than relying on R^2/duration alone.
MAX_PLAUSIBLE_VELOCITY_SEGMENTS_PER_S = 5.0


def summarize_window(lag_mat, corr_mat, min_corr=wga.MIN_PROPAGATION_CORR,
                      min_r2_for_velocity=MIN_R2_FOR_VELOCITY):
    """Derived per-window quantities from a 13x13 lag/corr matrix pair.

    - conduction_velocity_segments_per_s: slope^-1 of a lag-vs-distance
      linear fit across the upper-triangle pairs (i<j only, so distance
      and lag share one consistent direction convention). NaN unless
      r_squared >= min_r2_for_velocity -- see design note above; the raw,
      ungated value is always available as conduction_velocity_segments_per_s_raw.
    - r_squared: fit quality ("wave coherence") -- how well a single
      constant conduction velocity explains all 78 pairs at once. This is
      the number to trust; velocity is only meaningful conditional on this
      being reasonably high.
    - direction_consistency: fraction of pairs whose lag sign agrees with
      the fitted (majority) direction.
    - fit uses only pairs with |corr| >= min_corr ("strong" pairs, where
      corr here is the weakest adjacent link on the path between them --
      see pairwise_matrix_for_window); weak-link pairs would just add
      scatter inherited from a noisy segment-to-segment measurement.
    """
    n = lag_mat.shape[0]
    iu = np.triu_indices(n, k=1)
    distances = (iu[1] - iu[0]).astype(float)
    lags = lag_mat[iu]
    corrs = corr_mat[iu]

    strong = np.abs(corrs) >= min_corr
    n_strong = int(strong.sum())

    result = {
        "n_pairs_total": len(distances),
        "n_pairs_strong": n_strong,
        "mean_abs_corr": float(np.nanmean(np.abs(corrs))),
    }

    if n_strong < 5:
        result.update({
            "conduction_velocity_segments_per_s": np.nan,
            "conduction_velocity_segments_per_s_raw": np.nan,
            "velocity_reliable": False,
            "r_squared": np.nan,
            "direction_consistency": np.nan,
            "slope_s_per_segment": np.nan,
        })
        return result

    d, l = distances[strong], lags[strong]
    slope, intercept = np.polyfit(d, l, 1)
    pred = slope * d + intercept
    ss_res = np.sum((l - pred) ** 2)
    ss_tot = np.sum((l - np.mean(l)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    majority_sign = np.sign(slope) if slope != 0 else 0
    direction_consistency = float(np.mean(np.sign(l) == majority_sign)) if majority_sign != 0 else np.nan

    velocity_raw = (1.0 / slope) if slope != 0 else np.nan
    velocity_reliable = bool(
        not np.isnan(r_squared) and r_squared >= min_r2_for_velocity
        and not np.isnan(velocity_raw) and abs(velocity_raw) <= MAX_PLAUSIBLE_VELOCITY_SEGMENTS_PER_S
    )
    velocity = velocity_raw if velocity_reliable else np.nan

    result.update({
        "conduction_velocity_segments_per_s": velocity,
        "conduction_velocity_segments_per_s_raw": velocity_raw,
        "velocity_reliable": velocity_reliable,
        "slope_s_per_segment": slope,
        "r_squared": r_squared,
        "direction_consistency": direction_consistency,
    })
    return result


# Minimum window duration for the pairwise fit to be trusted at all,
# regardless of R^2. DESIGN NOTE (found 2026-08-26, right after adding the
# R^2 gate above): a bout shorter than the 10s target window produces a
# single short window (see sliding_windows_for_bout below) with very few
# samples -- one real example was a 0.6s/12-frame window that still
# produced R^2=0.71 and a nonphysical velocity of ~19 seg/s. A short,
# noisy fit can hit a high R^2 by chance; R^2 alone doesn't protect against
# that, only sample size does. Reuses wga.MIN_BOUT_SEC (3.0s) for
# consistency with the rest of the pipeline's bout-length convention
# rather than inventing a new threshold.
MIN_WINDOW_SEC_FOR_FIT = wga.MIN_BOUT_SEC

# |mean bodyAxisSpeed10| (this tracker's ~0-16 integer scale) at or below
# which a window is treated as "essentially not translating" for the
# r_squared default -- see the design note in analyze_file_pairwise where
# it's used. Compared by absolute value, not sign: 1 is the smallest
# non-zero magnitude on this scale (forward bouts require bodyAxisSpeed10 >
# 0 in every frame, so a mean at or below 1 means the window was forward in
# name only), and this stays correct if the threshold is ever reused
# somewhere signed speed can go negative (e.g. a reverse-locomotion bout).
NEAR_ZERO_SPEED_THRESHOLD = 1


def sliding_windows_for_bout(bs, be, fps=wga.FPS,
                              win_sec=wga.PROP_WINDOW_SEC, step_sec=wga.PROP_STEP_SEC):
    win = int(round(win_sec * fps))
    step = int(round(step_sec * fps))
    bout_len = be - bs
    if bout_len <= win:
        return [(bs, be)]
    windows = [(bs + s, bs + s + win) for s in range(0, bout_len - win + 1, step)]
    if windows[-1][1] < be:
        windows.append((be - win, be))
    return windows


def analyze_file_pairwise(path, fps=wga.FPS, cols=CURVATURE_COLS,
                           win_sec=wga.PROP_WINDOW_SEC, step_sec=wga.PROP_STEP_SEC,
                           min_r2_for_velocity=MIN_R2_FOR_VELOCITY):
    """Runs step 1 + step 2 over one file, restricted to forward-locomotion
    bouts (same bouts wga.analyze_file already uses), sliding-window over
    each bout. Returns (list_of_window_summary_dicts, filtered_df, mask).

    win_sec/step_sec override the pipeline-wide default sliding-window size
    (10s/5s) for this call only -- useful for checking how sensitive the
    coherence/velocity results are to that choice, without changing the
    default used everywhere else (the batch scripts, the neck/tail
    propagation heuristic in worm_gait_analysis.py, etc. are unaffected).
    Note MIN_WINDOW_SEC_FOR_FIT (3.0s) is NOT scaled with win_sec -- if
    win_sec is set below that floor, every window will fail the
    long-enough check, and r_squared/velocity will be NaN everywhere except
    windows where the worm was essentially stationary (see
    NEAR_ZERO_SPEED_THRESHOLD below), which get r_squared=0.

    min_r2_for_velocity overrides MIN_R2_FOR_VELOCITY for this call only.
    DESIGN NOTE (found 2026-08-26): MIN_R2_FOR_VELOCITY=0.5 was calibrated
    on 10s windows. Shortening win_sec to 5s (to better match a visual
    kymograph check) raised the rate of borderline-high velocity estimates
    passing the gate from 2/1334 (>5 seg/s, at 10s) to 23/1358 (>3 seg/s,
    at 5s) -- fewer samples per fit means a marginal R^2 clears the same
    threshold more often by chance. A shorter window needs a stricter R^2
    floor to end up equally trustworthy; this parameter is how a caller
    (e.g. a batch script using a non-default win_sec) supplies that without
    changing the default for every other caller of this function.
    """
    df = wga.load_full_data_table(path)
    filtered = filter_all_segments(df, cols=cols, fps=fps)
    mask = wga.forward_mask(df)
    bouts = wga.contiguous_runs(mask)

    rows = []
    for bout_idx, (bs, be) in enumerate(bouts):
        for (ws, we) in sliding_windows_for_bout(bs, be, fps=fps, win_sec=win_sec, step_sec=step_sec):
            lag_mat, corr_mat, adj_lag_sec, adj_corr, max_lag_sec = pairwise_matrix_for_window(
                filtered, ws, we, fps=fps, cols=cols
            )
            summary = summarize_window(lag_mat, corr_mat, min_r2_for_velocity=min_r2_for_velocity)

            window_duration_s = (we - ws) / fps
            long_enough = window_duration_s >= MIN_WINDOW_SEC_FOR_FIT
            if not long_enough:
                # Too few samples for the fit to be trustworthy regardless of
                # R^2 -- see MIN_WINDOW_SEC_FOR_FIT design note. Null out the
                # derived quantities rather than report a fit built on noise.
                summary.update({
                    "conduction_velocity_segments_per_s": np.nan,
                    "velocity_reliable": False,
                    "r_squared": np.nan,
                    "direction_consistency": np.nan,
                })

            # DESIGN NOTE (data owner, 2026-08-26): when r_squared couldn't
            # be computed at all (too few strong pairs, or window too short
            # -- both leave it NaN above), default it to 0 ONLY if the worm
            # was essentially not translating during this window (mean
            # bodyAxisSpeed10 <= NEAR_ZERO_SPEED_THRESHOLD) -- there, "no
            # wave coherence could be established" has a physical reason
            # (no real movement to correlate) and 0 is a meaningful value,
            # not a guess. If the window failed to fit despite real
            # movement (e.g. just too short, or noisy), r_squared stays NaN
            # -- we don't actually know what it would have been. This only
            # ever fills in a missing value; a real (non-NaN) r_squared from
            # summarize_window is never overwritten, even if speed happens
            # to be low too.
            #
            # r_squared_source records WHY r_squared has the value it does,
            # so downstream code (e.g. plotting) can tell these three cases
            # apart even though "fit" and "near_zero_speed_default" can both
            # produce a real, plottable number:
            #   "fit"                    - an actual pairwise fit (may still
            #                               be low, or fail velocity_reliable)
            #   "near_zero_speed_default" - no fit was possible, defaulted to
            #                               0 because the worm wasn't moving
            #   "unavailable"             - no fit was possible AND the worm
            #                               WAS moving -- r_squared is still
            #                               NaN; we genuinely don't know it
            mean_speed = df["bodyAxisSpeed10"].values[ws:we].mean()
            near_zero_speed = bool(abs(mean_speed) <= NEAR_ZERO_SPEED_THRESHOLD)
            if np.isnan(summary["r_squared"]):
                if near_zero_speed:
                    summary["r_squared"] = 0.0
                    r_squared_source = "near_zero_speed_default"
                else:
                    r_squared_source = "unavailable"
            else:
                r_squared_source = "fit"

            summary.update({
                "bout_idx": bout_idx, "bout_start": bs, "bout_end": be,
                "window_start": ws, "window_end": we,
                "window_start_s": ws / fps, "window_end_s": we / fps,
                "window_duration_s": window_duration_s, "window_long_enough": long_enough,
                "mean_body_axis_speed": mean_speed, "near_zero_speed": near_zero_speed,
                "r_squared_source": r_squared_source,
                "lag_mat": lag_mat, "corr_mat": corr_mat,
                "adjacent_lag_sec": adj_lag_sec, "adjacent_corr": adj_corr,
                "adjacent_search_range_sec": max_lag_sec,
                "n_weak_adjacent_links": int(np.sum(np.abs(adj_corr) < wga.MIN_PROPAGATION_CORR)),
            })
            rows.append(summary)
    return rows, filtered, mask

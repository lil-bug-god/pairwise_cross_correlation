"""
plot_pairwise_check.py

Step 2 verification: for one example file, runs the full pairwise
cross-correlation over sliding windows, then plots:
  (1) conduction velocity and wave-coherence (R^2) over the course of the
      recording, one point per window (the "over time" view),
  (2) a 13x13 lag heatmap + lag-vs-distance scatter/fit for the single BEST
      (highest-R^2) window,
  (3) the same heatmap+scatter view for the single WORST (lowest-R^2)
      window -- a look at what a low-coherence fit actually looks like,
      for comparison against (2).

Both (2) and (3) are restricted to windows with >=5 strong pairs (same
gate summarize_window uses before it will even compute a fit) -- "worst"
means worst REAL fit, not a window that was too sparse to fit at all.
(3) additionally excludes r_squared==0 exactly, which is summarize_window's
ss_tot==0 degenerate case (all strong-pair lags identical by coincidence),
not a genuine low-coherence measurement.

Usage:
    python plot_pairwise_check.py <path_to_fullDataTable.csv> [out_prefix]
        [--win-sec W] [--step-sec S] [--min-r2 R]

    --win-sec/--step-sec override the pipeline-default 10s/5s sliding window
    for this check only (same override body_wave_analysis.analyze_file_pairwise
    already exposes, and what run_batch_pairwise_analysis.py uses at 5s/2.5s
    for the batch run) -- pass the same values used for a batch run to
    cross-check this diagnostic against it; the two are otherwise
    independent and default differently.
    --step-sec defaults to half of --win-sec (50% overlap) if not given.
    --min-r2 defaults to 0.7 when --win-sec < 10s, else body_wave_analysis's
    own default (0.5) -- see MIN_R2_FOR_VELOCITY's design note for why a
    shorter window needs a stricter floor to stay equally trustworthy.
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import worm_gait_analysis as wga
import body_wave_analysis as bwa


def _plot_window_heatmap_and_scatter(path, window, out_path, descriptor):
    """13x13 lag heatmap + lag-vs-distance scatter/fit for one window dict
    (as returned by body_wave_analysis.analyze_file_pairwise). `descriptor`
    is a short label like 'best window (highest wave coherence)' or
    'worst window (lowest wave coherence)', used in the title only --
    shared by both the best- and worst-window plots since they're
    otherwise identical."""
    lag_mat, corr_mat = window["lag_mat"], window["corr_mat"]

    fig, (ax_heat, ax_scatter) = plt.subplots(1, 2, figsize=(13, 5.5))

    vlim = np.nanmax(np.abs(lag_mat))
    vlim = vlim if vlim > 0 else 1.0
    im = ax_heat.imshow(lag_mat, cmap="RdBu_r", vmin=-vlim, vmax=vlim, origin="upper")
    ax_heat.set_xticks(range(bwa.N_SEGMENTS))
    ax_heat.set_yticks(range(bwa.N_SEGMENTS))
    ax_heat.set_xticklabels(range(1, bwa.N_SEGMENTS + 1), fontsize=7)
    ax_heat.set_yticklabels(range(1, bwa.N_SEGMENTS + 1), fontsize=7)
    ax_heat.set_xlabel("Segment j")
    ax_heat.set_ylabel("Segment i")
    ax_heat.set_title("Lag(i,j), seconds\n(+ = j delayed relative to i)", fontsize=9)
    fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    n = bwa.N_SEGMENTS
    iu = np.triu_indices(n, k=1)
    distances = (iu[1] - iu[0]).astype(float)
    lags = lag_mat[iu]
    corrs = corr_mat[iu]
    strong = np.abs(corrs) >= wga.MIN_PROPAGATION_CORR

    ax_scatter.scatter(distances[~strong], lags[~strong], color="#cccccc", s=18, label="weak (|corr|<0.3)")
    ax_scatter.scatter(distances[strong], lags[strong], color="#2b6cb0", s=24, label="strong (|corr|>=0.3)")
    if not np.isnan(window["slope_s_per_segment"]):
        xs = np.array([distances.min(), distances.max()])
        ys = window["slope_s_per_segment"] * xs + np.polyfit(distances[strong], lags[strong], 1)[1]
        ax_scatter.plot(xs, ys, color="#c2610a", lw=1.5,
                         label=f"fit: R^2={window['r_squared']:.2f}, "
                               f"v={window['conduction_velocity_segments_per_s']:.2f} seg/s")
    ax_scatter.axhline(0, color="#888888", lw=0.75)
    ax_scatter.set_xlabel("Distance between segments (segment-index units)")
    ax_scatter.set_ylabel("Lag (s)")
    ax_scatter.legend(loc="best", fontsize=8, frameon=False)
    ax_scatter.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"{os.path.basename(path)}  |  {descriptor}: "
        f"t={window['window_start_s']:.1f}-{window['window_end_s']:.1f}s\n"
        f"adjacent-pair search range used: +/-{window['adjacent_search_range_sec']:.2f}s  |  "
        f"weak adjacent links: {window['n_weak_adjacent_links']}/12", fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_pairwise_check(path, out_prefix=None,
                         win_sec=wga.PROP_WINDOW_SEC, step_sec=wga.PROP_STEP_SEC,
                         min_r2_for_velocity=None):
    if min_r2_for_velocity is None:
        # Shorter-than-default windows need a stricter R^2 floor to stay
        # equally trustworthy -- see body_wave_analysis.MIN_R2_FOR_VELOCITY's
        # design note; 0.7 matches what run_batch_pairwise_analysis.py uses
        # for its 5s-window batch run.
        min_r2_for_velocity = 0.7 if win_sec < wga.PROP_WINDOW_SEC else bwa.MIN_R2_FOR_VELOCITY
    rows, filtered, mask = bwa.analyze_file_pairwise(
        path, win_sec=win_sec, step_sec=step_sec, min_r2_for_velocity=min_r2_for_velocity
    )
    if out_prefix is None:
        out_prefix = os.path.basename(path).replace(".csv", "")

    if not rows:
        print(f"No forward-locomotion windows found in {path}; nothing to plot.")
        return None

    fps = wga.FPS
    t_mid = np.array([(r["window_start_s"] + r["window_end_s"]) / 2 for r in rows])
    velocity = np.array([r["conduction_velocity_segments_per_s"] for r in rows])
    r2 = np.array([r["r_squared"] for r in rows])
    n_strong = np.array([r["n_pairs_strong"] for r in rows])

    # --- panel A: velocity & R^2 over time ---
    fig1, (ax_v, ax_r) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax_v.axhline(0, color="#888888", lw=0.75)
    ax_v.plot(t_mid, velocity, "o-", color="#2b6cb0", ms=4, lw=1)
    ax_v.set_ylabel(f"Conduction velocity\n(segments/s, R^2>={min_r2_for_velocity} only)")
    ax_v.spines[["top", "right"]].set_visible(False)

    sc = ax_r.scatter(t_mid, r2, c=n_strong, cmap="viridis", s=30, vmin=0, vmax=78)
    ax_r.axhline(0, color="#888888", lw=0.75)
    ax_r.set_ylim(-0.05, 1.05)
    ax_r.set_ylabel("Wave coherence (R^2)")
    ax_r.set_xlabel("Time (s)")
    ax_r.spines[["top", "right"]].set_visible(False)
    cbar = fig1.colorbar(sc, ax=ax_r, pad=0.01)
    cbar.set_label("n strong pairs (|corr| >= 0.3)\nof 78")

    fig1.suptitle(f"{os.path.basename(path)}  |  conduction velocity & wave coherence over time "
                  f"(one point per {win_sec:.0f}s window, {step_sec:.1f}s step, "
                  f"forward-locomotion bouts only)\n"
                  f"lags built from adjacent-pair narrow search + cumulative sum; velocity gaps = "
                  f"R^2 < {min_r2_for_velocity} (unreliable, 1/slope blows up near-zero slope)",
                  fontsize=10)
    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    out1 = f"{out_prefix}_pairwise_over_time.png"
    fig1.savefig(out1, dpi=150)
    plt.close(fig1)

    # --- pick the best/worst window (highest/lowest R^2), each tie-broken
    # by preferring MORE strong pairs (more evidence behind the number) ---
    # Filtered on n_pairs_strong (the actual "was a fit even possible"
    # condition), not on r_squared being non-NaN -- r_squared now defaults
    # to 0 rather than NaN when no fit was possible (see
    # body_wave_analysis.summarize_window's design note), so an isnan check
    # here would never exclude anything.
    valid = [i for i, r in enumerate(rows) if r["n_pairs_strong"] >= 5]
    if not valid:
        print(f"No window in {path} had >=5 strong pairs; skipping heatmap/scatter.")
        return out1, None, None
    best_i = max(valid, key=lambda i: (rows[i]["r_squared"], rows[i]["n_pairs_strong"]))

    # "Worst" additionally excludes r_squared==0 -- exactly 0 (rather than
    # just low) means the ss_tot==0 degenerate case in summarize_window (all
    # strong-pair lags identical by coincidence), not a real measurement of
    # poor coherence, so it isn't a representative "worst fit" to display.
    worst_candidates = [i for i in valid if rows[i]["r_squared"] != 0]
    if not worst_candidates:
        print(f"Every window in {path} with >=5 strong pairs had r_squared==0; skipping worst-window plot.")
        worst_i = None
    else:
        worst_i = min(worst_candidates, key=lambda i: (rows[i]["r_squared"], -rows[i]["n_pairs_strong"]))

    out2 = _plot_window_heatmap_and_scatter(
        path, rows[best_i], f"{out_prefix}_pairwise_best_window.png",
        "best window (highest wave coherence)",
    )
    out3 = None
    if worst_i is not None:
        out3 = _plot_window_heatmap_and_scatter(
            path, rows[worst_i], f"{out_prefix}_pairwise_worst_window.png",
            "worst window (lowest wave coherence)",
        )

    return out1, out2, out3


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="path to a fullDataTable.csv")
    parser.add_argument("out_prefix", nargs="?", default=None, help="output filename prefix (default: derived from input filename)")
    parser.add_argument("--win-sec", type=float, default=wga.PROP_WINDOW_SEC,
                         help=f"sliding-window size in seconds (default {wga.PROP_WINDOW_SEC:.0f}s)")
    parser.add_argument("--step-sec", type=float, default=None,
                         help="step between windows in seconds (default: half of --win-sec)")
    parser.add_argument("--min-r2", type=float, default=None,
                         help="R^2 threshold for a reliable velocity (default: 0.7 if --win-sec<10s, else 0.5)")
    args = parser.parse_args()
    step_sec = args.step_sec if args.step_sec is not None else args.win_sec / 2
    outs = plot_pairwise_check(
        args.path, out_prefix=args.out_prefix, win_sec=args.win_sec, step_sec=step_sec,
        min_r2_for_velocity=args.min_r2,
    )
    print(f"Saved {outs}")

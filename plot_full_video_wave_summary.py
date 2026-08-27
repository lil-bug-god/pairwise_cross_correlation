"""
plot_full_video_wave_summary.py

For ONE video, over its FULL duration (not a 2-minute clip, not a single
10s window): three time-aligned panels, stacked so you can visually trace
the same moment across all three --

  1. filtered curvature kymograph (all 13 segments, the actual signal the
     pairwise analysis runs on -- see body_wave_analysis.py)
  2. wave coherence (R^2) for every sliding window in the recording
  3. conduction velocity for every window that passed the reliability
     gates (R^2 >= 0.5, window >= 3s, |velocity| <= 5 seg/s -- see
     body_wave_analysis.MIN_R2_FOR_VELOCITY / MIN_WINDOW_SEC_FOR_FIT /
     MAX_PLAUSIBLE_VELOCITY_SEGMENTS_PER_S)

Panels 2 and 3 only have points where the video was in a forward-
locomotion bout -- gaps are real (turns, reversals, pauses), not missing
data. Points are connected only within a single bout (never across a gap
to the next bout) so the lines never imply continuity that isn't there.

Panel 2 distinguishes THREE reasons a point can be at/near R^2=0 (see
body_wave_analysis.analyze_file_pairwise's r_squared_source design note):
an actual fit that scored low ("velocity not reliable", open circle), a
window where no fit was possible but the worm was essentially stationary
so R^2=0 is a meaningful default (square marker), and a window where no
fit was possible despite real movement, so R^2 is genuinely unknown --
plotted as a light 'x' just below the axis, not at 0, since we don't
actually know its value.

Usage:
    python plot_full_video_wave_summary.py <path_to_fullDataTable.csv> [out.png]
        [--win-sec W] [--step-sec S] [--min-r2 R]

    --win-sec/--step-sec override the pipeline-default 10s/5s sliding window
    for this plot only (same override body_wave_analysis.analyze_file_pairwise
    already exposes, and what run_batch_pairwise_analysis.py uses at 5s/2.5s
    for the batch run) -- lets you compare window sizes on the same video.
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

RELIABLE_COLOR = "#2f8f5d"
UNRELIABLE_COLOR = "#999999"
NEAR_ZERO_COLOR = "#6b8fc2"
UNAVAILABLE_COLOR = "#cccccc"
VELOCITY_COLOR = "#c2610a"
BOUT_LINE_COLOR = "#888888"


def plot_full_video_summary(path, fps=wga.FPS, out_path=None,
                             win_sec=wga.PROP_WINDOW_SEC, step_sec=wga.PROP_STEP_SEC,
                             min_r2_for_velocity=None):
    if min_r2_for_velocity is None:
        # Shorter-than-default windows need a stricter R^2 floor to stay
        # equally trustworthy -- see body_wave_analysis.MIN_R2_FOR_VELOCITY's
        # design note; 0.7 matches what run_batch_pairwise_analysis.py uses
        # for its 5s-window batch run.
        min_r2_for_velocity = 0.7 if win_sec < wga.PROP_WINDOW_SEC else bwa.MIN_R2_FOR_VELOCITY
    rows, filtered, mask = bwa.analyze_file_pairwise(
        path, fps=fps, win_sec=win_sec, step_sec=step_sec, min_r2_for_velocity=min_r2_for_velocity
    )

    n_frames = len(filtered)
    t = np.arange(n_frames) / fps
    curv_cols = [f"curvature{i}" for i in range(1, bwa.N_SEGMENTS + 1)]
    curv = filtered[curv_cols].values.T  # (13, n_frames)

    fig, (ax_heat, ax_r2, ax_vel) = plt.subplots(
        3, 1, figsize=(16, 9), sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1, 1], "hspace": 0.12},
    )

    # --- panel 1: filtered curvature kymograph, full recording ---
    vlim = np.nanpercentile(np.abs(curv), 99)
    vlim = vlim if vlim > 0 else 1.0
    im = ax_heat.imshow(
        curv, aspect="auto", origin="upper", cmap="RdBu_r",
        vmin=-vlim, vmax=vlim, extent=[t[0], t[-1], bwa.N_SEGMENTS + 0.5, 0.5],
    )
    ax_heat.set_yticks([1, 2, 7, 13])
    ax_heat.set_yticklabels(["1 (head)", "2 (neck)", "7 (mid)", "13 (tail)"])
    ax_heat.set_ylabel("Body segment\n(curvature, filtered)")
    cbar = fig.colorbar(im, ax=ax_heat, pad=0.01, fraction=0.02)
    cbar.set_label("curvature (deg)")
    ax_heat.spines[["top", "right"]].set_visible(False)

    if not rows:
        for ax in (ax_r2, ax_vel):
            ax.text(0.5, 0.5, "No forward-locomotion windows found in this video",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10, color="#888888")
            ax.spines[["top", "right"]].set_visible(False)
        ax_vel.set_xlabel("Time (s)")
    else:
        t_mid = np.array([(r["window_start_s"] + r["window_end_s"]) / 2 for r in rows])
        r2 = np.array([r["r_squared"] for r in rows], dtype=float)
        vel = np.array([r["conduction_velocity_segments_per_s"] for r in rows], dtype=float)
        reliable = np.array([bool(r.get("velocity_reliable")) for r in rows])
        # r_squared can come from three different sources -- see
        # body_wave_analysis.analyze_file_pairwise's r_squared_source design
        # note -- which must be rendered differently: "fit" is a real
        # number (possibly low), "near_zero_speed_default" is a meaningful
        # 0 (worm wasn't moving), "unavailable" is still NaN (fit failed
        # despite real movement -- we don't know its value, so it can't be
        # plotted at its own r2 position at all).
        source = np.array([r.get("r_squared_source") for r in rows])
        fit_unreliable = (source == "fit") & ~reliable
        near_zero_default = source == "near_zero_speed_default"
        unavailable = source == "unavailable"
        bout_idx = np.array([r["bout_idx"] for r in rows])

        # --- panel 2: wave coherence, connected within a bout only ---
        # (NaN r2 for "unavailable" windows naturally breaks the connecting
        # line there, rather than interpolating through an unknown value.)
        for b in np.unique(bout_idx):
            m = bout_idx == b
            ax_r2.plot(t_mid[m], r2[m], color=BOUT_LINE_COLOR, lw=0.8, alpha=0.5, zorder=1)
        ax_r2.scatter(t_mid[reliable], r2[reliable], color=RELIABLE_COLOR, s=22, zorder=3,
                      label="velocity reliable")
        ax_r2.scatter(t_mid[fit_unreliable], r2[fit_unreliable], facecolors="none",
                      edgecolors=UNRELIABLE_COLOR, s=18, zorder=2, label="fit, velocity not reliable")
        ax_r2.scatter(t_mid[near_zero_default], r2[near_zero_default], marker="s", facecolors="none",
                      edgecolors=NEAR_ZERO_COLOR, s=22, zorder=2,
                      label=f"near-stationary (speed<={bwa.NEAR_ZERO_SPEED_THRESHOLD:.0f}), R^2=0")
        ax_r2.scatter(t_mid[unavailable], np.full(unavailable.sum(), -0.03), marker="x",
                      color=UNAVAILABLE_COLOR, s=16, zorder=2, label="fit failed, moving (R^2 unknown)")
        ax_r2.axhline(min_r2_for_velocity, color="#333333", lw=1, ls="--", alpha=0.7)
        ax_r2.text(t[-1], min_r2_for_velocity, f"  R^2={min_r2_for_velocity} threshold",
                   va="center", fontsize=8, color="#333333")
        ax_r2.set_ylabel("Wave coherence\n(R^2)")
        ax_r2.set_ylim(-0.05, 1.05)
        ax_r2.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
        ax_r2.spines[["top", "right"]].set_visible(False)

        # --- panel 3: conduction velocity, reliable windows only, connected within a bout ---
        any_reliable = False
        for b in np.unique(bout_idx):
            m = (bout_idx == b) & reliable
            if m.sum() > 0:
                ax_vel.plot(t_mid[m], vel[m], "o-", color=VELOCITY_COLOR, ms=4, lw=1.3, zorder=3)
                any_reliable = True
        if not any_reliable:
            ax_vel.text(0.5, 0.5, "No windows passed the velocity reliability gates in this video",
                        ha="center", va="center", transform=ax_vel.transAxes, fontsize=10, color="#888888")
        ax_vel.axhline(0, color="#888888", lw=0.75)
        ax_vel.set_ylabel("Conduction velocity\n(segments/s)")
        ax_vel.set_xlabel("Time (s)")
        ax_vel.spines[["top", "right"]].set_visible(False)

    filename = os.path.basename(path)
    n_windows = len(rows)
    n_reliable = int(np.sum([bool(r.get("velocity_reliable")) for r in rows])) if rows else 0
    fig.suptitle(
        f"{filename}  |  full-recording summary ({t[-1]:.0f}s, {n_frames} frames)\n"
        f"{win_sec:.0f}s window / {step_sec:.1f}s step  |  "
        f"{n_windows} windows evaluated, {n_reliable} with reliable velocity  |  "
        f"gaps in panels 2-3 = not in a forward-locomotion bout",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if out_path is None:
        out_path = filename.replace(".csv", "_full_video_summary.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="path to a fullDataTable.csv")
    parser.add_argument("out", nargs="?", default=None, help="output PNG path (default: <input>_full_video_summary.png)")
    parser.add_argument("--win-sec", type=float, default=wga.PROP_WINDOW_SEC,
                         help=f"sliding-window size in seconds (default {wga.PROP_WINDOW_SEC:.0f}s)")
    parser.add_argument("--step-sec", type=float, default=None,
                         help="step between windows in seconds (default: half of --win-sec)")
    parser.add_argument("--min-r2", type=float, default=None,
                         help="R^2 threshold for a reliable velocity (default: 0.7 if --win-sec<10s, else 0.5)")
    args = parser.parse_args()
    step_sec = args.step_sec if args.step_sec is not None else args.win_sec / 2
    out_path = plot_full_video_summary(
        args.path, out_path=args.out, win_sec=args.win_sec, step_sec=step_sec,
        min_r2_for_velocity=args.min_r2,
    )
    print(f"Saved {out_path}")

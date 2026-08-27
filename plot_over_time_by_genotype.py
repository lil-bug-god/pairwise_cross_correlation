"""
plot_over_time_by_genotype.py

Three aggregate-over-time views built from gait_pairwise_windows.csv:

1. Wave coherence (R^2) over time, all videos of a genotype overlaid --
   one panel per genotype, thin per-video points (each video keeps its own
   natural gaps -- it only has windows during its own forward-locomotion
   bouts, so points are not connected across long gaps) plus a bold
   binned, across-video mean trend line.

2. Wave coherence (R^2) over time, averaged across videos -- same binned
   trend lines as (1), but no individual points and both genotypes on one
   axes for direct comparison, matching (3)'s style.

3. Conduction velocity over time, averaged across videos -- one line per
   genotype (both on the same axes for direct comparison), built the same
   binned/across-video way, using only windows where velocity passed the
   R^2 and window-length gates (velocity_reliable == True).

Averaging convention (both plots): for each time bin, first take the mean
within each video (so a video that happens to contribute many windows to
a bin doesn't outweigh one that contributes few), THEN average those
per-video means across videos. This is the standard "average of subject
means" convention -- it weights every video equally regardless of how much
data it happened to contribute to a given bin.

Recordings vary a lot in length (190s to 930s in this dataset), so the
number of videos contributing to a bin shrinks at later times -- trend
lines are only drawn through bins with at least MIN_VIDEOS_PER_BIN
contributing videos, and the number of contributing videos is shown
directly so a viewer can't mistake a late-time wiggle (from 2-3 videos)
for the same kind of signal as an early-time one (from dozens).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Anchored to this script's own location, not the process's cwd -- was
# previously a sandbox-only path (/home/claude/...) that doesn't exist on a
# local machine. Matches run_batch_pairwise_analysis.py's WINDOWS_CSV, which
# writes here.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WINDOWS_CSV = os.path.join(SCRIPT_DIR, "gait_pairwise_windows.csv")
BIN_SEC = 30
MIN_VIDEOS_PER_BIN = 3

GENOTYPE_COLORS = {"N2": "#2b6cb0", "PHX9753": "#c2610a"}
GENOTYPE_ORDER = ["N2", "PHX9753"]


def add_time_bin(df, bin_sec=BIN_SEC):
    df = df.copy()
    df["window_mid_s"] = (df["window_start_s"] + df["window_end_s"]) / 2
    df["time_bin"] = (df["window_mid_s"] // bin_sec) * bin_sec
    return df


def binned_across_video_mean(df, value_col, bin_sec=BIN_SEC, min_videos=MIN_VIDEOS_PER_BIN):
    """Two-stage mean: per-file-per-bin mean, then mean (+ SEM, + n videos)
    across files, for each bin. Returns a DataFrame indexed by time_bin."""
    per_file_bin = (
        df.dropna(subset=[value_col])
        .groupby(["filename", "time_bin"])[value_col]
        .mean()
        .reset_index()
    )
    agg = per_file_bin.groupby("time_bin")[value_col].agg(
        mean="mean", sem=lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan,
        n_videos="count",
    )
    agg = agg.sort_index()
    agg["reliable_bin"] = agg["n_videos"] >= min_videos
    return agg


def plot_coherence_overlay(windows_df, out_path):
    df = add_time_bin(windows_df)
    # Derived from the data itself, not hardcoded -- this script is agnostic
    # to whatever win_sec the batch that produced windows_df actually used
    # (e.g. run_batch_pairwise_analysis.py currently uses 5s, not the
    # pipeline default of 10s; hardcoding "10s" here silently goes stale
    # the next time that changes).
    src_window_sec = windows_df["window_duration_s"].median()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True, sharex=True)

    for ax, genotype in zip(axes, GENOTYPE_ORDER):
        g = df[df["genotype"] == genotype]
        color = GENOTYPE_COLORS[genotype]

        # thin per-video points -- not connected, so bout gaps within a
        # video aren't misrepresented as continuous data
        for filename, gv in g.groupby("filename"):
            ax.scatter(gv["window_mid_s"], gv["r_squared"], color=color, s=8, alpha=0.15, linewidths=0)

        trend = binned_across_video_mean(g, "r_squared")
        reliable = trend[trend["reliable_bin"]]
        ax.plot(reliable.index + BIN_SEC / 2, reliable["mean"], color=color, lw=2.5, zorder=5)
        ax.fill_between(
            reliable.index + BIN_SEC / 2,
            reliable["mean"] - reliable["sem"], reliable["mean"] + reliable["sem"],
            color=color, alpha=0.25, zorder=4, linewidth=0,
        )
        unreliable = trend[~trend["reliable_bin"]]
        if len(unreliable):
            ax.plot(unreliable.index + BIN_SEC / 2, unreliable["mean"], color=color, lw=1, ls=":", alpha=0.5, zorder=3)

        n_videos_total = g["filename"].nunique()
        ax.set_title(f"{genotype}  (n={n_videos_total} videos)", fontsize=11)
        ax.set_xlabel("Time in recording (s)")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0, color="#888888", lw=0.75)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Wave coherence (R^2)")

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#888888", alpha=0.4,
                   markersize=6, label=f"individual video, one {src_window_sec:.0f}s window"),
        plt.Line2D([0], [0], color="#888888", lw=2.5, label=f"across-video mean (>= {MIN_VIDEOS_PER_BIN} videos in bin)"),
        plt.Line2D([0], [0], color="#888888", lw=1, ls=":", label=f"< {MIN_VIDEOS_PER_BIN} videos in bin (unreliable)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"Wave coherence over time, by genotype ({BIN_SEC:.0f}s bins, from {src_window_sec:.0f}s sliding windows)\n"
        f"faint dots = every {src_window_sec:.0f}s window from every video; bold line = mean across videos "
        f"(mean-of-video-means, shaded = SEM across videos)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.90])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_coherence_average(windows_df, out_path):
    """Same single-panel, both-genotypes-overlaid style as
    plot_velocity_average, but for wave coherence (R^2) -- no individual
    per-window dots, just the binned across-video mean trend lines, so the
    two genotypes can be compared directly on one axes. Unlike
    plot_velocity_average, this uses ALL windows (not just
    velocity_reliable ones): r_squared is a real number for essentially
    every window now (see body_wave_analysis's r_squared_source design
    note -- "fit" or "near_zero_speed_default"), it's only NaN for the
    rare "unavailable" case, which binned_across_video_mean already drops
    via dropna."""
    df = add_time_bin(windows_df)
    src_window_sec = windows_df["window_duration_s"].median()

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for genotype in GENOTYPE_ORDER:
        g = df[df["genotype"] == genotype]
        color = GENOTYPE_COLORS[genotype]
        trend = binned_across_video_mean(g, "r_squared")
        reliable = trend[trend["reliable_bin"]]
        unreliable = trend[~trend["reliable_bin"]]

        n_videos_total = g["filename"].nunique()
        ax.plot(reliable.index + BIN_SEC / 2, reliable["mean"], "o-", color=color, lw=2, ms=4,
                 label=f"{genotype} (n={n_videos_total} videos)", zorder=5)
        ax.fill_between(
            reliable.index + BIN_SEC / 2,
            reliable["mean"] - reliable["sem"], reliable["mean"] + reliable["sem"],
            color=color, alpha=0.2, zorder=4, linewidth=0,
        )
        if len(unreliable):
            ax.plot(unreliable.index + BIN_SEC / 2, unreliable["mean"], "o:", color=color, lw=1, ms=3, alpha=0.5, zorder=3)

    ax.axhline(0, color="#888888", lw=0.75)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time in recording (s)")
    ax.set_ylabel("Wave coherence (R^2)\nmean across videos")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="best", fontsize=9, frameon=False)

    fig.suptitle(
        f"Wave coherence over time, averaged across videos by genotype "
        f"({BIN_SEC:.0f}s bins, from {src_window_sec:.0f}s sliding windows)\n"
        f"mean-of-video-means, shaded = SEM across videos; "
        f"dotted segments = fewer than {MIN_VIDEOS_PER_BIN} videos contributing",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_velocity_average(windows_df, out_path):
    df = add_time_bin(windows_df)
    df_reliable = df[df["velocity_reliable"] == True]
    # Derived from the data, same reasoning as src_window_sec above: the
    # actual R^2 threshold used to set velocity_reliable can vary by batch
    # run (e.g. 0.7 for a 5s-window run vs. the pipeline default 0.5 for
    # 10s) -- inferring it from the reliable set's own minimum R^2 keeps
    # this label honest without needing to know which batch script/params
    # produced windows_df.
    src_r2_threshold = df_reliable["r_squared"].min() if len(df_reliable) else float("nan")
    src_window_sec = windows_df["window_duration_s"].median()

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for genotype in GENOTYPE_ORDER:
        g = df_reliable[df_reliable["genotype"] == genotype]
        color = GENOTYPE_COLORS[genotype]
        trend = binned_across_video_mean(g, "conduction_velocity_segments_per_s")
        reliable = trend[trend["reliable_bin"]]
        unreliable = trend[~trend["reliable_bin"]]

        n_videos_total = g["filename"].nunique()
        ax.plot(reliable.index + BIN_SEC / 2, reliable["mean"], "o-", color=color, lw=2, ms=4,
                 label=f"{genotype} (n={n_videos_total} videos)", zorder=5)
        ax.fill_between(
            reliable.index + BIN_SEC / 2,
            reliable["mean"] - reliable["sem"], reliable["mean"] + reliable["sem"],
            color=color, alpha=0.2, zorder=4, linewidth=0,
        )
        if len(unreliable):
            ax.plot(unreliable.index + BIN_SEC / 2, unreliable["mean"], "o:", color=color, lw=1, ms=3, alpha=0.5, zorder=3)

    ax.axhline(0, color="#888888", lw=0.75)
    ax.set_xlabel("Time in recording (s)")
    ax.set_ylabel("Conduction velocity (segments/s)\nmean across videos, reliable windows only")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="best", fontsize=9, frameon=False)

    fig.suptitle(
        f"Conduction velocity over time, averaged across videos by genotype "
        f"({BIN_SEC:.0f}s bins, from {src_window_sec:.0f}s sliding windows)\n"
        f"only windows with R^2>={src_r2_threshold:.2f} (and >=3s duration) are included; "
        f"dotted segments = fewer than {MIN_VIDEOS_PER_BIN} videos contributing",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    windows_df = pd.read_csv(WINDOWS_CSV)
    out1 = plot_coherence_overlay(windows_df, os.path.join(SCRIPT_DIR, "coherence_over_time_by_genotype.png"))
    out2 = plot_coherence_average(windows_df, os.path.join(SCRIPT_DIR, "coherence_average_over_time_by_genotype.png"))
    out3 = plot_velocity_average(windows_df, os.path.join(SCRIPT_DIR, "velocity_over_time_by_genotype.png"))
    print(out1)
    print(out2)
    print(out3)

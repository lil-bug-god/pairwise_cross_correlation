"""
run_batch_pairwise_analysis.py

Batch-runs body_wave_analysis.analyze_file_pairwise() over every
fullDataTable.csv under the staging-method/genotype directory tree,
producing:

  gait_pairwise_windows.csv   -- one row per sliding window, every file
                                  (the finest-grained output; lag_mat/corr_mat
                                  matrices themselves are not included, only
                                  the derived scalar summary fields)
  gait_pairwise_by_file.csv   -- one row per file, aggregated across its
                                  windows
  gait_pairwise_by_group.csv  -- one row per staging_method x genotype group

Resumable: if gait_pairwise_windows.csv already exists, files already
present in it are skipped, so a partial/interrupted run can continue
without redoing completed files.
"""

import os
import re
import glob
import time
import numpy as np
import pandas as pd

import worm_gait_analysis as wga
import body_wave_analysis as bwa

# Anchored to this script's own location (not the process's cwd) so this
# works whether it's run via VS Code's "Run Python File" button, the
# integrated terminal, or any other cwd -- was previously a sandbox-only
# path (/mnt/user-data/uploads/...) that doesn't exist on a local machine.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, "..", "..", "fullDataTables_diffStagingMethods", "0coverslip_spacers")
WINDOWS_CSV = os.path.join(SCRIPT_DIR, "gait_pairwise_windows.csv")
BY_FILE_CSV = os.path.join(SCRIPT_DIR, "gait_pairwise_by_file.csv")
BY_GROUP_CSV = os.path.join(SCRIPT_DIR, "gait_pairwise_by_group.csv")

# Sliding-window size for THIS (pairwise conduction velocity / wave
# coherence) analysis only. Chosen 2026-08-26 after comparing 5s/10s/20s
# windows side-by-side against the filtered curvature heatmap for one
# video (see plot_full_video_wave_summary.py) -- 5s tracked the visible
# wave structure best. Deliberately set here, at the call site, rather
# than by changing wga.PROP_WINDOW_SEC/PROP_STEP_SEC (10s/5s): those two
# constants are also the window used by the separate neck/tail propagation
# heuristic in worm_gait_analysis.py, which underlies the frequency and
# first/second-half comparisons -- changing them globally would silently
# change that analysis too, which was validated at 10s and isn't part of
# this request. Step is kept at half the window (50% overlap), matching
# the original 10s/5s ratio.
PAIRWISE_WINDOW_SEC = 5
PAIRWISE_STEP_SEC = 2.5

# Tightened from the pipeline default (0.5) for this 5s-window run only.
# DESIGN NOTE (found 2026-08-26): the 0.5 threshold was calibrated on 10s
# windows. At 5s, fewer samples back each fit, so the same threshold let
# more borderline-high velocity estimates through by chance (23/1358 over
# 3 seg/s, several near the 5 seg/s plausibility ceiling, vs. 2/1334 at
# 10s). Checked empirically across thresholds -- 0.7 brings the tail back
# to parity with the accepted 10s behavior (2 windows over 3 seg/s) while
# still keeping 720 reliable windows, comparable to the 704 at 10s/R^2>=0.5.
PAIRWISE_MIN_R2_FOR_VELOCITY = 0.7

DATE_RE = re.compile(r"(\d{8})")
CAM_RE = re.compile(r"cam(\d+)", re.IGNORECASE)
REP_RE = re.compile(r"_(\d{3})_fullDataTable\.csv$", re.IGNORECASE)

WINDOW_FIELDS = [
    "window_start_s", "window_end_s", "window_duration_s", "window_long_enough",
    "mean_body_axis_speed", "near_zero_speed", "r_squared_source",
    "bout_idx", "n_pairs_total", "n_pairs_strong", "mean_abs_corr",
    "n_weak_adjacent_links", "adjacent_search_range_sec",
    "conduction_velocity_segments_per_s", "conduction_velocity_segments_per_s_raw",
    "velocity_reliable", "slope_s_per_segment", "r_squared", "direction_consistency",
]


def agg_file(g):
    reliable = g[g["velocity_reliable"] == True]
    return pd.Series({
        "n_windows": len(g),
        "n_windows_long_enough": int(g["window_long_enough"].sum()),
        "n_windows_reliable_velocity": len(reliable),
        "frac_windows_reliable_velocity": len(reliable) / len(g) if len(g) else np.nan,
        "mean_r_squared": g["r_squared"].mean(),
        "median_r_squared": g["r_squared"].median(),
        "mean_velocity_segments_per_s": reliable["conduction_velocity_segments_per_s"].mean(),
        "median_velocity_segments_per_s": reliable["conduction_velocity_segments_per_s"].median(),
        "std_velocity_segments_per_s": reliable["conduction_velocity_segments_per_s"].std(),
        "mean_direction_consistency": g["direction_consistency"].mean(),
    })


def parse_metadata(path):
    parts = path.split(os.sep)
    filename = parts[-1]
    genotype = parts[-2]
    staging_method = parts[-3]
    date_m = DATE_RE.search(filename)
    cam_m = CAM_RE.search(filename)
    rep_m = REP_RE.search(filename)
    return {
        "staging_method": staging_method,
        "genotype": genotype,
        "date": date_m.group(1) if date_m else None,
        "camera": f"cam{cam_m.group(1)}" if cam_m else None,
        "replicate": rep_m.group(1) if rep_m else None,
        "fed": "fed" in filename.lower(),
        "filename": filename,
    }


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "**", "*.csv"), recursive=True))
    print(f"Found {len(files)} CSV files under {ROOT}")

    already_done = set()
    existing_rows = []
    if os.path.exists(WINDOWS_CSV):
        prev = pd.read_csv(WINDOWS_CSV)
        already_done = set(prev["filename"].unique())
        existing_rows = prev.to_dict("records")
        print(f"Resuming: {len(already_done)} files already done, skipping those.")

    all_window_rows = list(existing_rows)
    file_errors = {}

    t0 = time.time()
    for i, path in enumerate(files, 1):
        meta = parse_metadata(path)
        if meta["filename"] in already_done:
            continue
        try:
            rows, filtered, mask = bwa.analyze_file_pairwise(
                path, win_sec=PAIRWISE_WINDOW_SEC, step_sec=PAIRWISE_STEP_SEC,
                min_r2_for_velocity=PAIRWISE_MIN_R2_FOR_VELOCITY,
            )
        except Exception as exc:
            file_errors[meta["filename"]] = str(exc)
            print(f"[{i}/{len(files)}] {meta['filename']}: ERROR {exc}")
            continue

        n_reliable = sum(1 for r in rows if r.get("velocity_reliable"))
        for r in rows:
            row = {**meta, **{k: r.get(k) for k in WINDOW_FIELDS}}
            all_window_rows.append(row)

        elapsed = time.time() - t0
        print(f"[{i}/{len(files)}] {meta['staging_method']}/{meta['genotype']}/{meta['filename']}: "
              f"{len(rows)} windows, {n_reliable} with reliable velocity  "
              f"(elapsed {elapsed:.0f}s)")

        # write incrementally so a partial run isn't lost
        pd.DataFrame(all_window_rows).to_csv(WINDOWS_CSV, index=False)

    windows_df = pd.DataFrame(all_window_rows)
    windows_df.to_csv(WINDOWS_CSV, index=False)
    print(f"\nWrote {len(windows_df)} window-rows across "
          f"{windows_df['filename'].nunique()} files to {WINDOWS_CSV}")
    if file_errors:
        print(f"\n{len(file_errors)} files errored and were skipped entirely:")
        for fn, err in file_errors.items():
            print(f"  {fn}: {err}")

    # --- per-file aggregation ---
    by_file = (
        windows_df.groupby(["staging_method", "genotype", "filename"])
        .apply(agg_file, include_groups=False)
        .reset_index()
    )
    by_file.to_csv(BY_FILE_CSV, index=False)
    print(f"Wrote {len(by_file)} rows to {BY_FILE_CSV}")

    # --- group-level aggregation ---
    by_group = by_file.groupby(["staging_method", "genotype"]).agg(
        n_files=("filename", "count"),
        mean_frac_windows_reliable=("frac_windows_reliable_velocity", "mean"),
        mean_r_squared=("mean_r_squared", "mean"),
        mean_velocity_segments_per_s=("mean_velocity_segments_per_s", "mean"),
        median_velocity_segments_per_s=("median_velocity_segments_per_s", "median"),
        mean_direction_consistency=("mean_direction_consistency", "mean"),
    )
    print("\n--- Summary by staging_method x genotype ---")
    print(by_group)
    by_group.to_csv(BY_GROUP_CSV)
    print(f"\nWrote group summary to {BY_GROUP_CSV}")

    total_elapsed = time.time() - t0
    print(f"\nTotal elapsed: {total_elapsed:.0f}s")


if __name__ == "__main__":
    main()

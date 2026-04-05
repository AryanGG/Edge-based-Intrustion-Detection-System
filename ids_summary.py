# =============================================================================
# ids_summary.py — Intrusion Event Log Viewer
# =============================================================================
# Run:  python ids_summary.py
#
# Reads intrusion_log.csv (or the path set in config.py) and prints a
# formatted summary of all past intrusion events.
# =============================================================================

from __future__ import annotations

import os
import sys

import pandas as pd

import config


def main() -> None:
    log_path = config.LOG_FILE

    if not os.path.isfile(log_path):
        print(f"[INFO] No log file found at '{log_path}'.")
        print("       Start the main detection system first — events are logged automatically.")
        return

    df = pd.read_csv(log_path, parse_dates=["timestamp"])

    if df.empty:
        print("[INFO] Log file exists but contains no intrusion events yet.")
        return

    # ── Summary header ────────────────────────────────────────────────────────
    total   = len(df)
    first_t = df["timestamp"].min()
    last_t  = df["timestamp"].max()
    avg_c   = df["confidence"].mean()
    max_c   = df["confidence"].max()

    separator = "─" * 72

    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║          Edge-Based Intrusion Detection System — Event Log           ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    print(f"  {'Total intrusion events':<30} {total}")
    print(f"  {'First event':<30} {first_t}")
    print(f"  {'Last event':<30} {last_t}")
    print(f"  {'Avg confidence':<30} {avg_c:.4f}")
    print(f"  {'Max confidence':<30} {max_c:.4f}")
    print()
    print(separator)

    # ── Per-event table ───────────────────────────────────────────────────────
    print(f"  {'#':<5}  {'Timestamp':<27}  {'Conf':>6}  Snapshot")
    print(separator)

    for i, row in df.iterrows():
        ts   = row["timestamp"]
        conf = row["confidence"]
        snap = row["snapshot_path"]
        snap_display = snap if len(str(snap)) <= 45 else "…" + str(snap)[-42:]
        print(f"  {int(i)+1:<5}  {str(ts):<27}  {conf:>6.4f}  {snap_display}")

    print(separator)
    print()

    # ── Snapshot existence check ───────────────────────────────────────────────
    missing = [r["snapshot_path"] for _, r in df.iterrows()
               if not os.path.isfile(str(r["snapshot_path"]))]
    if missing:
        print(f"  [WARN] {len(missing)} snapshot file(s) in the log are missing from disk.")
    else:
        print(f"  [OK]   All {total} snapshot file(s) are present on disk.")

    print()


if __name__ == "__main__":
    main()

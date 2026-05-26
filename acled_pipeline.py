"""
ACLED conflict data pipeline — SEIHRF-OD model
Lancet Infectious Diseases submission, 2026

Downloads ACLED events for eastern DRC (Ituri, Nord-Kivu, Sud-Kivu),
aggregates into a weekly conflict-intensity time series C(t), and
produces the CSV file expected by the Stan model.

Usage:
    1. Register at https://acleddata.com and obtain a free API key.
    2. Set ACLED_EMAIL and ACLED_KEY below (or as environment variables).
    3. Run:  python acled_pipeline.py

Outputs:
    - data/acled_Ct_weekly.csv      (C(t) series for Stan)
    - figS3_acled_Ct.pdf            (Supplementary Figure S3)

Dependencies:
    pip install requests pandas numpy matplotlib

ACLED API reference: https://apidocs.acleddata.com
"""

import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# ── Configuration ──────────────────────────────────────────────────────────────

ACLED_EMAIL = os.environ.get("ACLED_EMAIL", "YOUR_EMAIL@example.com")
ACLED_KEY   = os.environ.get("ACLED_KEY",   "YOUR_API_KEY_HERE")

# Outbreak window: index case 24 April 2026 → SitRep 007 = 20 May 2026
# Extend to 90 days for the projection window
START_DATE  = "2026-04-01"   # buffer before index case
END_DATE    = "2026-07-31"   # covers 90-day projection

# Eastern DRC provinces of interest
PROVINCES = ["Ituri", "Nord-Kivu", "Sud-Kivu"]

# ACLED event types associated with armed conflict
# See: https://acleddata.com/resources/general-guides/
CONFLICT_EVENT_TYPES = [
    "Battles",
    "Violence against civilians",
    "Explosions/Remote violence",
]

# Outbreak start (day 0 = 24 April 2026)
OUTBREAK_START = datetime(2026, 4, 24)

# Output paths
OUTPUT_CT_CSV  = "data/acled_Ct_weekly.csv"
OUTPUT_FIG     = "figS3_acled_Ct.pdf"

# ── ACLED download ─────────────────────────────────────────────────────────────

ACLED_API_URL = "https://api.acleddata.com/acled/read"

def download_acled(start: str, end: str, provinces: list) -> pd.DataFrame:
    """
    Download all armed-conflict events for eastern DRC from the ACLED API.
    Returns a DataFrame with one row per event.
    """
    if ACLED_EMAIL == "YOUR_EMAIL@example.com" or ACLED_KEY == "YOUR_API_KEY_HERE":
        raise ValueError(
            "Set ACLED_EMAIL and ACLED_KEY before running.\n"
            "Register free at https://acleddata.com/register/"
        )

    all_events = []
    page = 1
    page_size = 500

    print("[acled_pipeline] Downloading ACLED events...")
    while True:
        params = {
            "key"        : ACLED_KEY,
            "email"      : ACLED_EMAIL,
            "country"    : "Democratic Republic of Congo",
            "admin1"     : "|".join(provinces),
            "event_date" : f"{start}|{end}",
            "event_date_where": "BETWEEN",
            "event_type" : "|".join(CONFLICT_EVENT_TYPES),
            "fields"     : "event_date|admin1|admin2|event_type|fatalities|geo_precision",
            "page"       : page,
            "limit"      : page_size,
        }
        resp = requests.get(ACLED_API_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "data" not in data or not data["data"]:
            break

        all_events.extend(data["data"])
        print(f"  Page {page}: {len(data['data'])} events downloaded")

        if len(data["data"]) < page_size:
            break
        page += 1

    if not all_events:
        raise RuntimeError(
            "No ACLED events returned. Check date range, province names, and API credentials."
        )

    df = pd.DataFrame(all_events)
    df["event_date"] = pd.to_datetime(df["event_date"], format="%Y-%m-%d")
    df["fatalities"] = pd.to_numeric(df["fatalities"], errors="coerce").fillna(0)
    print(f"[acled_pipeline] Downloaded {len(df)} events total.")
    return df


# ── Conflict-intensity aggregation ────────────────────────────────────────────

def compute_Ct(df: pd.DataFrame,
               start: datetime,
               end: datetime,
               method: str = "event_count") -> pd.DataFrame:
    """
    Aggregate ACLED events into a weekly conflict-intensity C(t) series.

    method: "event_count"  — number of conflict events per week (default)
            "fatalities"   — total fatalities per week
            "combined"     — 0.5 * normalized(events) + 0.5 * normalized(fatalities)

    Returns a DataFrame with columns: week_start, day, C_raw, C_normalized.
    """
    # Build weekly bins
    freq = "W-MON"   # week starting Monday
    weeks = pd.date_range(start=start, end=end, freq=freq)

    records = []
    for w in weeks:
        w_end = w + timedelta(days=6)
        mask  = (df["event_date"] >= w) & (df["event_date"] <= w_end)
        subset = df[mask]

        if method == "event_count":
            intensity = len(subset)
        elif method == "fatalities":
            intensity = subset["fatalities"].sum()
        else:   # combined
            intensity = 0.5 * len(subset) + 0.5 * subset["fatalities"].sum()

        # Day index relative to outbreak start
        day = (w - OUTBREAK_START).days

        records.append({
            "week_start"  : w.strftime("%Y-%m-%d"),
            "day"         : day,
            "C_raw"       : intensity,
        })

    result = pd.DataFrame(records)

    # Normalize to [0, 1] so C(t) ∈ [0,1] as expected by the Stan model
    c_max = result["C_raw"].max()
    if c_max > 0:
        result["C_normalized"] = result["C_raw"] / c_max
    else:
        result["C_normalized"] = 0.0

    return result


def compare_with_stepfunction(ct_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the documented-event C(t) used in the paper.

    Five anchors (all events with direct documented impact on Ebola response):
      day ≤  0  C = 0.30  Pre-epidemic insecurity — OCHA Ituri Q1 2026
                           (32 600 displaced; 11 incidents vs humanitarian actors)
                           cited in WHO DON602
      day 1–16  C = 0.30  Ongoing access constraints (DON602)
      day 17    C = 0.55  Accidental exposure of US health worker at Nyankunde
                           Hospital, Bunia, 11 May 2026 (CDC; The Guardian)
      day 24    C = 0.65  CDC public announcement + medical evacuation to Berlin,
                           18 May 2026 — high-visibility community-trust event (CDC)
      day 27–29 C = 1.00  Peak cluster: Rwampara tent burnings (21 May);
                           Ituri hospital material burned, WHO DG: "significantly
                           compromised" (22 May); Mongbwalu — 18 patients fled
                           treatment centre (23 May)  [Le Devoir; DON603; Al Jazeera]
      day 30+   C = 0.60  Persistent insecurity — >100 000 displaced, Ituri &
                           Nord-Kivu (OCHA May 2026; DON603)
    """
    day_col = ct_df["day"].values

    # Piecewise anchors: (start_day, end_day_inclusive, magnitude)
    anchors = [
        (None,  0,  0.30),   # pre-epidemic background (days ≤ 0)
        (1,    16,  0.30),   # ongoing access constraints
        (17,   23,  0.55),   # Nyankunde exposure + escalation
        (24,   26,  0.65),   # CDC announcement + evacuation to Berlin
        (27,   29,  1.00),   # Rwampara / Ituri / Mongbwalu peak cluster
        (30,  None, 0.60),   # persistent post-incident insecurity
    ]

    step = np.zeros(len(day_col))
    for start, end, mag in anchors:
        for j, d in enumerate(day_col):
            after_start = (start is None) or (d >= start)
            before_end  = (end   is None) or (d <= end)
            if after_start and before_end:
                step[j] = mag

    ct_df = ct_df.copy()
    ct_df["C_step"] = step
    return ct_df


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_Ct(ct_df: pd.DataFrame, output_path: str):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                             gridspec_kw={"hspace": 0.35})

    days = ct_df["day"].values

    # Panel A: raw event count
    ax = axes[0]
    ax.bar(days, ct_df["C_raw"], width=6, color="#8B0000", alpha=0.75,
           label="Weekly conflict events (ACLED)")
    ax.set_ylabel("Event count", fontsize=10)
    ax.set_title("(A)  Raw ACLED conflict events — Ituri, Nord-Kivu, Sud-Kivu",
                 fontsize=10, loc="left")
    ax.legend(fontsize=9, frameon=False)

    # Mark outbreak milestones
    for day, label, color in [
        (0,  "Index case\n(24 Apr)",     "#2E7D32"),
        (17, "Nyankunde\nexposure\n(11 May)", "#E65100"),
        (24, "CDC announ.\nevacuation\n(18 May)", "#6A1B9A"),
        (27, "Rwampara\n(21 May)",        "#B71C1C"),
        (29, "Mongbwalu\n(23 May)",       "#B71C1C"),
    ]:
        ax.axvline(day, color=color, linestyle="--", linewidth=1, alpha=0.8)
        ax.text(day + 0.5, ax.get_ylim()[1] * 0.85, label,
                fontsize=7.5, color=color, va="top")

    # Panel B: normalized C(t) vs original step function
    ax = axes[1]
    ax.step(days, ct_df["C_normalized"], where="post",
            color="#004E7D", linewidth=2, label="ACLED-derived $C(t)$")
    if "C_step" in ct_df.columns:
        ax.step(days, ct_df["C_step"], where="post",
                color="gray", linewidth=1.5, linestyle="--",
                label="Original step-function (paper v3)")
    ax.set_xlabel("Day since index case (24 April 2026)", fontsize=10)
    ax.set_ylabel("Normalized $C(t)$ ∈ [0, 1]", fontsize=10)
    ax.set_title("(B)  Conflict intensity $C(t)$ — ACLED vs. step-function approximation",
                 fontsize=10, loc="left")
    ax.legend(fontsize=9, frameon=False)
    ax.set_ylim(-0.05, 1.15)

    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[acled_pipeline] Figure saved → {output_path}")


# ── Stan integration check ────────────────────────────────────────────────────

def export_for_stan(ct_df: pd.DataFrame, output_path: str):
    """
    Export the C(t) series in a format directly readable by the Stan model.
    Columns: day (integer, from 0), C (normalized, [0,1]).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    export = ct_df[["day", "C_normalized"]].rename(columns={"C_normalized": "C"})
    export.to_csv(output_path, index=False)
    print(f"[acled_pipeline] Stan-ready C(t) saved → {output_path}")
    print(f"  {len(export)} weekly data points; max C = {export['C'].max():.3f}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ACLED conflict pipeline — SEIHRF-OD model")
    print("=" * 60)

    # Download
    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_dt   = datetime.strptime(END_DATE,   "%Y-%m-%d")
    df_raw   = download_acled(START_DATE, END_DATE, PROVINCES)

    # Save raw download for reproducibility
    os.makedirs("data", exist_ok=True)
    raw_path = "data/acled_raw.csv"
    df_raw.to_csv(raw_path, index=False)
    print(f"[acled_pipeline] Raw events saved → {raw_path}")

    # Compute C(t)
    ct_df = compute_Ct(df_raw, start_dt, end_dt, method="event_count")

    # Add step-function for comparison
    ct_df = compare_with_stepfunction(ct_df)

    # Export for Stan
    export_for_stan(ct_df, OUTPUT_CT_CSV)

    # Plot
    plot_Ct(ct_df, OUTPUT_FIG)

    # Summary statistics
    n_events = df_raw.groupby(
        pd.Grouper(key="event_date", freq="W")
    ).size()
    high_weeks = (n_events > n_events.mean() + n_events.std()).sum()

    print("\n── Summary ──────────────────────────────────────────────────")
    print(f"  Total ACLED events downloaded : {len(df_raw)}")
    print(f"  Date range covered            : {START_DATE} → {END_DATE}")
    print(f"  Provinces                     : {', '.join(PROVINCES)}")
    print(f"  High-conflict weeks (>mean+1σ): {high_weeks}")
    print(f"  Max normalized C(t)           : {ct_df['C_normalized'].max():.3f}")
    print("─────────────────────────────────────────────────────────────")
    print("\nNext steps:")
    print("  1. Inspect figS3_acled_Ct.pdf and compare with step-function.")
    print("  2. Update your Stan model to read C(t) from data/acled_Ct_weekly.csv.")
    print("  3. Recalibrate and compare posterior with original step-function run.")
    print("  4. If conclusions are unchanged (expected), replace the step-function")
    print("     paragraph in the Methods section with the ACLED-derived results.")


if __name__ == "__main__":
    main()

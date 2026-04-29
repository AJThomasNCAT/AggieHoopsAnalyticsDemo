"""
build_team_stats_json.py
------------------------
Reads player_advanced.csv (produced by feature_engineering.py) and writes a
clean JSON file the website can use to replace the hardcoded "Games 1-4"
dataset on the Team Stats tab.

Output: assets/data/team_stats_2025_26.json

The structure mirrors what the existing chart code expects:

{
  "name":     "Men's Season Averages (2025-26)",
  "season":   "2025-26",
  "games":    30,
  "last_updated": "2026-04-25",
  "players":  [
    {"#": "00", "Player": "Walker, Lureon",
     "GP":   29,
     "AVG_MIN": 35.1, "AVG_PTS": 18.9, "AVG_REB": 6.0, "AVG_AST": 2.2,
     "AVG_STL": 1.1,  "AVG_BLK": 0.3,  "AVG_TO":  2.4,
     "FG_PCT": 0.452, "FG3_PCT": 0.378, "FT_PCT": 0.812,
     "TS_PCT": 0.642, "EFG_PCT": 0.521, "USG_PCT": 24.5,
     "AVG_PTS_PER40": 21.5, "AVG_REB_PER40": 6.8, "AVG_AST_PER40": 2.5},
    ...
  ],
  "stats": ["AVG_PTS", "AVG_REB", "AVG_AST", "AVG_MIN",
            "AVG_STL", "AVG_BLK", "AVG_TO",
            "FG_PCT", "FG3_PCT", "FT_PCT",
            "TS_PCT", "EFG_PCT", "USG_PCT",
            "AVG_PTS_PER40", "AVG_REB_PER40", "AVG_AST_PER40"]
}

Run after feature_engineering.py:
    python3 build_team_stats_json.py
"""

import pandas as pd
import json
import os
from datetime import datetime

OUT_DIR     = "assets/data"
INPUT_PATH  = os.path.join(OUT_DIR, "player_advanced.csv")
OUTPUT_PATH = os.path.join(OUT_DIR, "team_stats_2025_26.json")

if not os.path.exists(INPUT_PATH):
    raise SystemExit(f"\n{INPUT_PATH} not found. Run feature_engineering.py first.\n")

print("Loading player_advanced.csv...")
df = pd.read_csv(INPUT_PATH, parse_dates=["game_date"])

# ── 1. Filter to NC A&T 2025-26 ───────────────────────────────────────────────
# season=2026 is the 2025-26 season (named by year-end convention)
# We keep ALL games (including <10 min) so GP and totals are accurate, but
# rate stats are meaningless for very short stints.

ncat = df[
    df["team"].str.contains("A&T", na=False) &
    (df["season"] == 2026)
].copy()

print(f"NC A&T 2025-26 player-game rows: {len(ncat):,}")
print(f"Games covered:                   {ncat['game_id'].nunique()}")
print(f"Unique players:                  {ncat['player'].nunique()}\n")


# ── 2. Calculate per-player season aggregates ────────────────────────────────
# Volume stats we sum then divide by GP for averages.
# Rate stats (FG%, TS%, etc.) we sum the underlying makes/attempts then divide
# — averaging-of-averages is wrong for percentages.

agg = (ncat.groupby("player")
            .agg(
                GP   = ("game_id", "count"),
                MIN  = ("min", "sum"),
                PTS  = ("pts", "sum"),
                REB  = ("reb", "sum"),
                AST  = ("ast", "sum"),
                STL  = ("stl", "sum"),
                BLK  = ("blk", "sum"),
                TO   = ("to",  "sum"),
                FGM  = ("fgm", "sum"),
                FGA  = ("fga", "sum"),
                FG3M = ("fg3m", "sum"),
                FG3A = ("fg3a", "sum"),
                FTM  = ("ftm", "sum"),
                FTA  = ("fta", "sum"),
                # We need a representative jersey number — pick the most common one
                # (it should be constant across games but pandas needs an aggregator)
                JERSEY = ("position", "first"),  # placeholder; we'll fix below
            )
            .reset_index())


# ── 3. Compute averages ──────────────────────────────────────────────────────
# Per-game averages — divide totals by GP
for col in ["MIN", "PTS", "REB", "AST", "STL", "BLK", "TO"]:
    agg[f"AVG_{col}"] = (agg[col] / agg["GP"]).round(1)

# Shooting percentages — properly weighted (sum of makes / sum of attempts)
def safe_pct(numerator, denominator):
    return ((numerator / denominator).fillna(0).round(3))

agg["FG_PCT"]  = safe_pct(agg["FGM"],  agg["FGA"])
agg["FG3_PCT"] = safe_pct(agg["FG3M"], agg["FG3A"])
agg["FT_PCT"]  = safe_pct(agg["FTM"],  agg["FTA"])

# eFG% — gives 1.5x credit for 3-pointers since they're worth 1.5x
agg["EFG_PCT"] = safe_pct(agg["FGM"] + 0.5 * agg["FG3M"], agg["FGA"])

# True Shooting % — accounts for FT efficiency too
agg["TS_PCT"]  = safe_pct(agg["PTS"], 2 * (agg["FGA"] + 0.44 * agg["FTA"]))


# ── 4. Per-40 stats and Usage Rate (already computed in player_advanced.csv) ──
# We average them at the player level. This is OK because they're already rates.
extras = (ncat.groupby("player")
              .agg(
                  AVG_PTS_PER40 = ("pts_per40", "mean"),
                  AVG_REB_PER40 = ("reb_per40", "mean"),
                  AVG_AST_PER40 = ("ast_per40", "mean"),
                  AVG_USG_PCT   = ("usg_pct",   "mean"),
              )
              .round(1)
              .reset_index())

agg = agg.merge(extras, on="player", how="left")
agg = agg.rename(columns={"AVG_USG_PCT": "USG_PCT"})

# Round USG_PCT and per-40s to 1 decimal
for col in ["USG_PCT", "AVG_PTS_PER40", "AVG_REB_PER40", "AVG_AST_PER40"]:
    agg[col] = agg[col].round(1)


# ── 5. Build player records ──────────────────────────────────────────────────
# We don't have jersey numbers in the scraped data, so we leave them blank.
# (The chart code uses the Player column anyway for the labels.)

# Sort by AVG_PTS descending so leading scorers appear first
agg = agg.sort_values("AVG_PTS", ascending=False).reset_index(drop=True)

players = []
for _, row in agg.iterrows():
    players.append({
        "#":              "",
        "Player":         row["player"],
        "GP":             int(row["GP"]),
        "AVG_MIN":        float(row["AVG_MIN"]),
        "AVG_PTS":        float(row["AVG_PTS"]),
        "AVG_REB":        float(row["AVG_REB"]),
        "AVG_AST":        float(row["AVG_AST"]),
        "AVG_STL":        float(row["AVG_STL"]),
        "AVG_BLK":        float(row["AVG_BLK"]),
        "AVG_TO":         float(row["AVG_TO"]),
        "FG_PCT":         float(row["FG_PCT"]),
        "FG3_PCT":        float(row["FG3_PCT"]),
        "FT_PCT":         float(row["FT_PCT"]),
        "EFG_PCT":        float(row["EFG_PCT"]),
        "TS_PCT":         float(row["TS_PCT"]),
        "USG_PCT":        float(row["USG_PCT"]),
        "AVG_PTS_PER40":  float(row["AVG_PTS_PER40"]),
        "AVG_REB_PER40":  float(row["AVG_REB_PER40"]),
        "AVG_AST_PER40":  float(row["AVG_AST_PER40"]),
    })


# ── 6. Build full dataset object ─────────────────────────────────────────────
output = {
    "name":   "Men's Season Averages (2025-26)",
    "season": "2025-26",
    "games":  int(ncat["game_id"].nunique()),
    "last_updated": datetime.now().strftime("%Y-%m-%d"),
    "players": players,
    "stats": [
        # Per-game volume averages
        "AVG_PTS", "AVG_REB", "AVG_AST", "AVG_MIN",
        "AVG_STL", "AVG_BLK", "AVG_TO",
        # Shooting percentages
        "FG_PCT", "FG3_PCT", "FT_PCT",
        # Advanced efficiency
        "TS_PCT", "EFG_PCT", "USG_PCT",
        # Per-40 (the advisor-friendly stats)
        "AVG_PTS_PER40", "AVG_REB_PER40", "AVG_AST_PER40",
    ],
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)


# ── 7. Summary ────────────────────────────────────────────────────────────────
print(f"✅  Saved to {OUTPUT_PATH}")
print(f"\nPlayers (sorted by avg pts):")
for p in players:
    pct_str = f" ({p['FG_PCT']*100:.0f}% FG)" if p["AVG_MIN"] > 5 else ""
    print(f"  {p['Player']:<25}  GP {p['GP']:>2}  "
          f"{p['AVG_PTS']:>5.1f} pts  {p['AVG_REB']:>4.1f} reb  "
          f"{p['AVG_AST']:>4.1f} ast{pct_str}")
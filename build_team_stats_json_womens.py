"""
build_team_stats_json_womens.py
--------------------------------
Reads player_advanced_womens.csv and writes team_stats_2025_26_womens.json
for the Women's Season Averages section of the Team Stats tab.

Run:
    python3 build_team_stats_json_womens.py
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

OUT_DIR     = "assets/data"
INPUT_PATH  = os.path.join(OUT_DIR, "player_advanced_womens.csv")
OUTPUT_PATH = os.path.join(OUT_DIR, "team_stats_2025_26_womens.json")

if not os.path.exists(INPUT_PATH):
    raise SystemExit(
        f"\n{INPUT_PATH} not found.\n"
        f"Run python3 feature_engineering_womens.py first.\n"
    )

print("Loading player_advanced_womens.csv...")
df = pd.read_csv(INPUT_PATH, parse_dates=["game_date"])

# Filter to NC A&T women's 2025-26 season
ncat = df[
    df["team"].str.contains("A&T", na=False) &
    (df["season"] == 2026)
].copy()

print(f"NC A&T women's 2025-26 rows: {len(ncat):,}")
print(f"Games covered:               {ncat['game_id'].nunique()}")
print(f"Unique players:              {ncat['player'].nunique()}\n")

# Aggregate per player
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
           )
           .reset_index())

# Per-game averages
for col in ["MIN", "PTS", "REB", "AST", "STL", "BLK", "TO"]:
    agg[f"AVG_{col}"] = (agg[col] / agg["GP"]).round(1)

safe = lambda n, d: (n / d).fillna(0).round(3)

agg["FG_PCT"]  = safe(agg["FGM"],  agg["FGA"])
agg["FG3_PCT"] = safe(agg["FG3M"], agg["FG3A"])
agg["FT_PCT"]  = safe(agg["FTM"],  agg["FTA"])
agg["EFG_PCT"] = safe(agg["FGM"] + 0.5 * agg["FG3M"], agg["FGA"])
agg["TS_PCT"]  = safe(agg["PTS"],  2 * (agg["FGA"] + 0.44 * agg["FTA"]))

# Per-40 and usage from advanced CSV
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
agg = agg.sort_values("AVG_PTS", ascending=False).reset_index(drop=True)

# Only include players with 3+ games and 5+ avg minutes
agg = agg[(agg["GP"] >= 3) & (agg["AVG_MIN"] >= 5)]

players = []
for _, row in agg.iterrows():
    players.append({
        "#":             "",
        "Player":        row["player"],
        "GP":            int(row["GP"]),
        "AVG_MIN":       float(row["AVG_MIN"]),
        "AVG_PTS":       float(row["AVG_PTS"]),
        "AVG_REB":       float(row["AVG_REB"]),
        "AVG_AST":       float(row["AVG_AST"]),
        "AVG_STL":       float(row["AVG_STL"]),
        "AVG_BLK":       float(row["AVG_BLK"]),
        "AVG_TO":        float(row["AVG_TO"]),
        "FG_PCT":        float(row["FG_PCT"]),
        "FG3_PCT":       float(row["FG3_PCT"]),
        "FT_PCT":        float(row["FT_PCT"]),
        "EFG_PCT":       float(row["EFG_PCT"]),
        "TS_PCT":        float(row["TS_PCT"]),
        "USG_PCT":       float(row["USG_PCT"]),
        "AVG_PTS_PER40": float(row["AVG_PTS_PER40"]),
        "AVG_REB_PER40": float(row["AVG_REB_PER40"]),
        "AVG_AST_PER40": float(row["AVG_AST_PER40"]),
    })

output = {
    "name":   "Women's Season Averages (2025-26)",
    "gender": "womens",
    "season": "2025-26",
    "games":  int(ncat["game_id"].nunique()),
    "last_updated": datetime.now().strftime("%Y-%m-%d"),
    "players": players,
    "stats": [
        "AVG_PTS", "AVG_REB", "AVG_AST", "AVG_MIN",
        "AVG_STL", "AVG_BLK", "AVG_TO",
        "FG_PCT", "FG3_PCT", "FT_PCT",
        "TS_PCT", "EFG_PCT", "USG_PCT",
        "AVG_PTS_PER40", "AVG_REB_PER40", "AVG_AST_PER40",
    ],
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"✅  Saved to {OUTPUT_PATH}")
print(f"\nPlayers (sorted by avg pts):")
for p in players:
    pct = f" ({p['FG_PCT']*100:.0f}% FG)" if p["AVG_MIN"] > 5 else ""
    print(f"  {p['Player']:<25}  GP {p['GP']:>2}  "
          f"{p['AVG_PTS']:>5.1f} pts  "
          f"{p['AVG_REB']:>4.1f} reb  "
          f"{p['AVG_AST']:>4.1f} ast{pct}")

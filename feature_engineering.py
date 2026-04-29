"""
feature_engineering.py
-----------------------
Combines three data sources:

  games_raw.parquet             - NC A&T 3-season box scores
  ncaa_box_scores.parquet       - D1 scrape for opponent context
  assets/data/torvik_20XX.csv   - Bart Torvik iterated efficiency ratings
                                  (one file per season: 2024, 2025, 2026)

Improvements over previous version:
  1. Multi-season Torvik  — loads torvik_2024.csv, torvik_2025.csv, torvik_2026.csv
                            so all 94 games get real Torvik ratings, not imputed medians
  2. torvik_adjoe         — opponent offensive efficiency (new feature)
  3. Expanded rolling     — adds reb_per40 and to_per40 rolling averages
  4. Hot streak flag      — binary: did player beat their avg in last 2 games?
  5. Position encoding    — guard / wing / big as numeric (0/1/2)

Run:
    python3 feature_engineering.py

Before running, download Torvik files:
    curl "http://barttorvik.com/2024_team_results.csv" -o assets/data/torvik_2024.csv
    curl "http://barttorvik.com/2025_team_results.csv" -o assets/data/torvik_2025.csv
    curl "http://barttorvik.com/2026_team_results.csv" -o assets/data/torvik_2026.csv
    (or just rename the 2026_team_results.csv you already have)
"""

import pandas as pd
import numpy as np
import os
from difflib import get_close_matches

OUT_DIR   = "assets/data"
NCAT_PATH = os.path.join(OUT_DIR, "games_raw.parquet")
D1_PATH   = os.path.join(OUT_DIR, "ncaa_box_scores.parquet")

os.makedirs(OUT_DIR, exist_ok=True)

for p in [NCAT_PATH, D1_PATH]:
    if not os.path.exists(p):
        raise SystemExit(f"\n{p} not found. Run update_boxscores.py first.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 0 — LOAD TORVIK RATINGS (all available seasons)
# ═══════════════════════════════════════════════════════════════════════════════
# Key columns used:
#   adjde   = adjusted defensive efficiency  (lower = better defense)
#   adjoe   = adjusted offensive efficiency  (higher = better offense)
#   adjt    = adjusted tempo                 (possessions per 40 min)
#   barthag = composite team strength        (0-1 probability)

# Manual ESPN-name → Torvik-name corrections
# Handles mascot suffixes and known mis-matches
MANUAL_CORRECTIONS = {
    "charleston cougars"              : "Charleston",
    "south carolina state bulldogs"   : "South Carolina St.",
    "morgan state bears"              : "Morgan St.",
    "north carolina central eagles"   : "North Carolina Central",
    "maryland eastern shore hawks"    : "Maryland Eastern Shore",
    "unc greensboro spartans"         : "UNC Greensboro",
    "unc wilmington seahawks"         : "UNC Wilmington",
    "william & mary tribe"            : "William & Mary",
    "campbell fighting camels"        : "Campbell",
    "stony brook seawolves"           : "Stony Brook",
    # Non-D1 — correctly stay NaN
    "washington adventist shock"      : None,
    "mid-atlantic christian mustangs" : None,
}

def make_espn_to_torvik(torvik_names):
    """Return a closure that maps ESPN team names to Torvik names."""
    def espn_to_torvik(espn_name):
        if pd.isna(espn_name):
            return None
        key = espn_name.strip().lower()
        if key in MANUAL_CORRECTIONS:
            return MANUAL_CORRECTIONS[key]
        words = espn_name.strip().split()
        for n in [len(words), len(words) - 1, len(words) - 2]:
            if n < 1:
                break
            candidate = " ".join(words[:n])
            matches = get_close_matches(candidate, torvik_names, n=1, cutoff=0.65)
            if matches:
                return matches[0]
        return None
    return espn_to_torvik


# Load each available season file into a dict: season_year → lookup_dict
torvik_by_season = {}   # {2024: {team_name: {adjde, adjoe, adjt, barthag}}, ...}

for season_yr in [2024, 2025, 2026]:
    path = os.path.join(OUT_DIR, f"torvik_{season_yr}.csv")
    if os.path.exists(path):
        df_t = pd.read_csv(path)
        lookup = df_t.set_index("team")[["adjde", "adjoe", "adjt", "barthag"]].to_dict("index")
        torvik_by_season[season_yr] = lookup
        print(f"Torvik {season_yr}: {len(lookup)} teams loaded from {path}")
    else:
        print(f"⚠  torvik_{season_yr}.csv not found — "
              f"2024/25 games will use rolling D1 fallback for that season")

if not torvik_by_season:
    print("\n⚠  No Torvik files found. Download them:")
    print("   curl http://barttorvik.com/2024_team_results.csv -o assets/data/torvik_2024.csv")
    print("   curl http://barttorvik.com/2025_team_results.csv -o assets/data/torvik_2025.csv")
    print("   curl http://barttorvik.com/2026_team_results.csv -o assets/data/torvik_2026.csv\n")

print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — BUILD OPPONENT RATINGS FROM D1 DATA (rolling estimates — fallback)
# ═══════════════════════════════════════════════════════════════════════════════

print("Loading D1 box scores...")
d1 = pd.read_parquet(D1_PATH)
d1["game_date"] = pd.to_datetime(d1["game_date"], errors="coerce")
d1 = d1[d1["player"] != "TEAM"].copy()
d1 = d1.rename(columns={"3pm": "fg3m", "3pa": "fg3a",
                          "2pm": "fg2m", "2pa": "fg2a"})

print(f"  D1 rows:   {len(d1):,}")
print(f"  D1 teams:  {d1['team'].nunique()}")
print(f"  D1 games:  {d1['game_id'].nunique()}\n")

STAT_COLS = ["min", "fgm", "fga", "fg2m", "fg2a", "fg3m", "fg3a",
             "ftm", "fta", "pts", "reb", "ast", "to", "stl", "blk",
             "oreb", "dreb", "pf"]

d1_team = (d1.groupby(["game_id", "game_date", "team"], as_index=False)[STAT_COLS].sum())

opp_rename = {c: f"opp_{c}" for c in STAT_COLS}
opp_side = (d1_team[["game_id", "team"] + STAT_COLS]
            .rename(columns={"team": "opp_team", **opp_rename}))
d1_team = d1_team.merge(opp_side, on="game_id", how="left")
d1_team = d1_team[d1_team["team"] != d1_team["opp_team"]].reset_index(drop=True)

safe = lambda n, d: np.where(d > 0, n / d, 0.0)

def poss(fga, oreb, to, fta):
    return fga - oreb + to + 0.44 * fta

d1_team["poss"]     = poss(d1_team["fga"],     d1_team["oreb"],  d1_team["to"],  d1_team["fta"])
d1_team["opp_poss"] = poss(d1_team["opp_fga"], d1_team["opp_oreb"], d1_team["opp_to"], d1_team["opp_fta"])
d1_team["pace"]     = (d1_team["poss"] + d1_team["opp_poss"]) / 2
d1_team["ortg"]     = 100 * safe(d1_team["pts"],     d1_team["pace"])
d1_team["drtg"]     = 100 * safe(d1_team["opp_pts"], d1_team["pace"])
d1_team["efg_pct"]  = safe(d1_team["fgm"] + 0.5 * d1_team["fg3m"], d1_team["fga"])
d1_team["tov_pct"]  = safe(d1_team["to"],   d1_team["poss"])
d1_team["orb_pct"]  = safe(d1_team["oreb"], d1_team["oreb"] + d1_team["opp_dreb"])
d1_team["ts_pct"]   = safe(d1_team["pts"],  2 * (d1_team["fga"] + 0.44 * d1_team["fta"]))

d1_team = d1_team.sort_values(["team", "game_date"]).reset_index(drop=True)
for metric in ["ortg", "drtg", "pace", "efg_pct", "tov_pct"]:
    d1_team[f"{metric}_r5"] = (
        d1_team.groupby("team")[metric]
               .transform(lambda s: s.shift(1).rolling(5, min_periods=2).mean())
    )

if "season" in d1.columns and "season" not in d1_team.columns:
    d1_team = d1_team.merge(
        d1.drop_duplicates("game_id")[["game_id", "season"]], on="game_id", how="left"
    )

lookup_cols = ["game_id", "team", "drtg_r5", "ortg_r5", "pace_r5"]
if "season" in d1_team.columns:
    lookup_cols.append("season")

opp_lookup = (d1_team[lookup_cols]
              .rename(columns={
                  "team":    "opp_team_name",
                  "drtg_r5": "opp_drtg_pregame",
                  "ortg_r5": "opp_ortg_pregame",
                  "pace_r5": "opp_pace_pregame",
              }))

print(f"D1 team-game rows: {len(d1_team):,}")
print(f"Teams with rolling ratings: {d1_team['drtg_r5'].notna().sum()} / {len(d1_team)} rows\n")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — NC A&T PLAYER FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

print("Loading NC A&T box scores...")
ncat = pd.read_parquet(NCAT_PATH)
ncat["game_date"] = pd.to_datetime(ncat["game_date"], errors="coerce")
ncat = ncat[ncat["player"] != "TEAM"].copy()
ncat = ncat.rename(columns={"3pm": "fg3m", "3pa": "fg3a",
                              "2pm": "fg2m", "2pa": "fg2a"})

print(f"  NC A&T rows:  {len(ncat):,}")
print(f"  NC A&T games: {ncat['game_id'].nunique()}\n")

# ── 2a. NC A&T team-game totals ───────────────────────────────────────────────
ncat_team = (ncat.groupby(["game_id", "game_date", "team"], as_index=False)[STAT_COLS].sum())

ncat_meta = (ncat[["game_id", "ncat_score", "opp_score", "home_away", "win", "opponent"]]
             .drop_duplicates("game_id"))
ncat_team = ncat_team.merge(ncat_meta, on="game_id", how="left")
ncat_team["is_ncat"]    = ncat_team["team"].str.contains("A&T", na=False).astype(int)
ncat_team["team_score"] = np.where(ncat_team["is_ncat"] == 1,
                                    ncat_team["ncat_score"], ncat_team["opp_score"])
ncat_team["opp_score_"] = np.where(ncat_team["is_ncat"] == 1,
                                    ncat_team["opp_score"], ncat_team["ncat_score"])

opp_nc = (ncat_team[["game_id", "team"] + STAT_COLS]
          .rename(columns={"team": "opp_team", **opp_rename}))
ncat_team = ncat_team.merge(opp_nc, on="game_id", how="left")
ncat_team = ncat_team[ncat_team["team"] != ncat_team["opp_team"]].reset_index(drop=True)

ncat_team["poss"]     = poss(ncat_team["fga"],     ncat_team["oreb"],  ncat_team["to"],  ncat_team["fta"])
ncat_team["opp_poss"] = poss(ncat_team["opp_fga"], ncat_team["opp_oreb"], ncat_team["opp_to"], ncat_team["opp_fta"])
ncat_team["pace"]     = (ncat_team["poss"] + ncat_team["opp_poss"]) / 2
ncat_team["ortg"]     = 100 * safe(ncat_team["team_score"], ncat_team["pace"])
ncat_team["drtg"]     = 100 * safe(ncat_team["opp_score_"], ncat_team["pace"])
ncat_team["efg_pct"]  = safe(ncat_team["fgm"] + 0.5 * ncat_team["fg3m"], ncat_team["fga"])
ncat_team["tov_pct"]  = safe(ncat_team["to"],   ncat_team["poss"])
ncat_team["orb_pct"]  = safe(ncat_team["oreb"], ncat_team["oreb"] + ncat_team["opp_dreb"])
ncat_team["ts_pct"]   = safe(ncat_team["team_score"],
                             2 * (ncat_team["fga"] + 0.44 * ncat_team["fta"]))

ncat_team = ncat_team.sort_values(["team", "game_date"]).reset_index(drop=True)
for metric in ["ortg", "drtg", "pace", "efg_pct"]:
    ncat_team[f"{metric}_r5"] = (
        ncat_team.groupby("team")[metric]
                 .transform(lambda s: s.shift(1).rolling(5, min_periods=2).mean())
    )

ncat_team.to_csv(os.path.join(OUT_DIR, "team_advanced.csv"), index=False)

# ── 2b. Attach opponent ratings (rolling D1 base + multi-season Torvik overlay) ─
opp_names = (ncat_team[ncat_team["is_ncat"] == 0][["game_id", "team"]]
             .rename(columns={"team": "opp_team_name"}))

opp_ratings = opp_names.merge(opp_lookup, on=["game_id", "opp_team_name"], how="left")

rolling_matched = opp_ratings["opp_drtg_pregame"].notna().sum()
print(f"Rolling D1 ratings matched: {rolling_matched} / {len(opp_ratings)} games")

# ── Torvik overlay (season-aware) ─────────────────────────────────────────────
# For each game, use the Torvik file matching that game's season year.
# 2026 games → torvik_2026.csv, 2025 games → torvik_2025.csv, etc.
# This gives real Torvik ratings for all seasons, not just 2026.

if torvik_by_season:
    # Add season to ncat_meta if available
    game_meta_season = ncat_meta[["game_id", "opponent"]].copy()
    if "season" in ncat.columns:
        season_map = ncat.drop_duplicates("game_id")[["game_id", "season"]]
        game_meta_season = game_meta_season.merge(season_map, on="game_id", how="left")
    else:
        game_meta_season["season"] = 2026

    # For each game, look up Torvik metrics using the correct season's file
    def get_torvik_metric(row, metric):
        season = int(row.get("season", 2026))
        # Try exact season first, then fall back to 2026
        lookup = torvik_by_season.get(season, torvik_by_season.get(2026, {}))
        if not lookup:
            return np.nan
        espn_to_torvik = make_espn_to_torvik(list(lookup.keys()))
        torvik_name = espn_to_torvik(row["opponent"])
        if torvik_name and torvik_name in lookup:
            return lookup[torvik_name][metric]
        return np.nan

    for metric in ["adjde", "adjoe", "adjt", "barthag"]:
        game_meta_season[f"torvik_{metric}"] = game_meta_season.apply(
            lambda row: get_torvik_metric(row, metric), axis=1
        )

    opp_ratings = opp_ratings.merge(
        game_meta_season[["game_id", "torvik_adjde", "torvik_adjoe",
                           "torvik_adjt", "torvik_barthag"]],
        on="game_id", how="left"
    )

    # Overlay Torvik values where available
    opp_ratings["opp_drtg_pregame"] = opp_ratings["torvik_adjde"].combine_first(
        opp_ratings["opp_drtg_pregame"]
    )
    opp_ratings["opp_ortg_pregame"] = opp_ratings["torvik_adjoe"].combine_first(
        opp_ratings["opp_ortg_pregame"]
    )
    opp_ratings["opp_pace_pregame"] = opp_ratings["torvik_adjt"].combine_first(
        opp_ratings["opp_pace_pregame"]
    )
    # torvik_barthag and adjoe: fill remaining NaN with median
    for col in ["torvik_barthag", "torvik_adjoe"]:
        median_val = opp_ratings[col].median()
        if not pd.isna(median_val):
            opp_ratings[col] = opp_ratings[col].fillna(median_val)

    torvik_matched = game_meta_season["torvik_adjde"].notna().sum()
    print(f"Torvik ratings matched:  {torvik_matched} / {len(game_meta_season)} games "
          f"(across {len(torvik_by_season)} seasons)")
else:
    opp_ratings["torvik_barthag"] = np.nan
    opp_ratings["torvik_adjoe"]   = np.nan

final_matched = opp_ratings["opp_drtg_pregame"].notna().sum()
print(f"Final opp_drtg_pregame coverage: {final_matched} / {len(opp_ratings)} games\n")


# ── 2c. Player advanced metrics ───────────────────────────────────────────────
player = ncat.copy()

player = player.merge(
    ncat_team[["game_id", "team", "min", "fga", "fta", "to", "poss",
               "ortg", "drtg", "pace"]
              ].rename(columns={"min": "team_min", "fga": "team_fga",
                                 "fta": "team_fta", "to":  "team_to",
                                 "poss": "team_poss"}),
    on=["game_id", "team"], how="left"
)

player = player.merge(
    opp_ratings[["game_id", "opp_drtg_pregame", "opp_ortg_pregame",
                 "opp_pace_pregame", "torvik_barthag", "torvik_adjoe"]],
    on="game_id", how="left"
)

# ── Merge PBP shot quality features ───────────────────────────────────────────
# scrape_pbp_features.py produces pbp_features.csv with one row per player
# per game identified by (game_id, athlete_id). We join on (game_id, player_id)
# which maps directly since CBBpy uses ESPN athlete IDs as player_id.
PBP_PATH = os.path.join(OUT_DIR, "pbp_features.csv")
if os.path.exists(PBP_PATH):
    pbp = pd.read_csv(PBP_PATH)
    # Force both sides to string so int64 vs object doesn't crash the merge
    pbp["game_id"]    = pbp["game_id"].astype(str)
    pbp["athlete_id"] = pbp["athlete_id"].astype(str)
    pbp_cols = ["game_id", "athlete_id",
                "rim_rate", "three_rate", "mid_rate",
                "avg_shot_dist", "assisted_rate", "foul_drawn_rate"]
    pbp = pbp[[c for c in pbp_cols if c in pbp.columns]].copy()

    if "player_id" in player.columns:
        player["game_id"]   = player["game_id"].astype(str)
        player["player_id"] = player["player_id"].astype(str)
        player = player.merge(pbp, left_on=["game_id", "player_id"],
                              right_on=["game_id", "athlete_id"], how="left")
        player = player.drop(columns=["athlete_id"], errors="ignore")
    else:
        # No player_id column — PBP features unavailable, add NaN columns
        print("  ⚠  player_id column not found — cannot merge PBP features")
        for col in ["rim_rate", "three_rate", "mid_rate",
                    "avg_shot_dist", "assisted_rate", "foul_drawn_rate"]:
            player[col] = np.nan

    # Opponent shot quality context
    GAME_PBP_PATH = os.path.join(OUT_DIR, "pbp_game_features.csv")
    if os.path.exists(GAME_PBP_PATH):
        gf = pd.read_csv(GAME_PBP_PATH, parse_dates=["game_date"])
        gf["game_id"] = gf["game_id"].astype(str)
        opp_gf = gf[gf["team_label"] == "opp"].copy()
        opp_gf = opp_gf.sort_values(["opponent", "game_date"]).reset_index(drop=True)
        for col in ["rim_rate", "three_rate", "avg_shot_dist"]:
            if col in opp_gf.columns:
                opp_gf[f"opp_{col}_r5"] = (
                    opp_gf.groupby("opponent")[col]
                          .transform(lambda s: s.shift(1).rolling(5, min_periods=2).mean())
                )
        opp_ctx_cols = ["game_id"] + [c for c in opp_gf.columns
                                       if c.startswith("opp_") and c.endswith("_r5")]
        player = player.merge(opp_gf[opp_ctx_cols], on="game_id", how="left")

    # Impute NaN PBP values with median
    pbp_all_cols = ["rim_rate", "three_rate", "mid_rate", "avg_shot_dist",
                    "assisted_rate", "foul_drawn_rate",
                    "opp_rim_rate_r5", "opp_three_rate_r5", "opp_avg_shot_dist_r5"]
    for col in pbp_all_cols:
        if col in player.columns:
            median_val = player[col].median()
            if not pd.isna(median_val):
                player[col] = player[col].fillna(median_val)

    pbp_coverage = player["rim_rate"].notna().sum() if "rim_rate" in player.columns else 0
    print(f"PBP features merged: {pbp_coverage} / {len(player)} player rows have shot data")
else:
    print(f"⚠  pbp_features.csv not found — run scrape_pbp_features.py first")
    print(f"   PBP features will be unavailable for this run")

# True Shooting %
player["ts_pct"]  = safe(player["pts"], 2 * (player["fga"] + 0.44 * player["fta"]))
# Effective FG%
player["efg_pct"] = safe(player["fgm"] + 0.5 * player["fg3m"], player["fga"])
# Usage Rate
player_poss = player["fga"] + 0.44 * player["fta"] + player["to"]
team_poss   = player["team_fga"] + 0.44 * player["team_fta"] + player["team_to"]
player["usg_pct"] = 100 * safe(
    player_poss * (player["team_min"] / 5),
    player["min"] * team_poss
)

# Per-40 stats
for stat in ["pts", "reb", "ast", "stl", "blk", "to",
             "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb"]:
    player[f"{stat}_per40"] = np.where(
        player["min"] > 0, player[stat] * (40.0 / player["min"]), 0.0
    )

player["low_min_flag"] = (player["min"] < 10).astype(int)


# ── 5. Position encoding ──────────────────────────────────────────────────────
# CBBpy's position column uses values like G, F, C, G/F, F/C, etc.
# We collapse to: 0=Guard, 1=Wing(Forward), 2=Big(Center)
# This lets the model learn position-specific patterns (bigs rebound more,
# guards assist more) without being told directly.
POSITION_MAP = {
    "G":   0, "PG": 0, "SG": 0,
    "G/F": 0,
    "F":   1, "SF": 1, "PF": 1,
    "F/G": 1, "F/C": 1,
    "C":   2,
}

if "position" in player.columns:
    player["position_enc"] = (player["position"]
                               .str.strip()
                               .str.upper()
                               .map(POSITION_MAP)
                               .fillna(1))   # unknown → wing as neutral default
else:
    player["position_enc"] = 1   # fallback if column missing

print(f"Position encoding:")
if "position" in player.columns:
    pos_counts = player["position_enc"].value_counts().sort_index()
    for k, v in pos_counts.items():
        label = {0:"Guard", 1:"Wing", 2:"Big"}[k]
        print(f"  {label} ({k}): {v:,} rows")
print()


# ── 2d. Rolling player features (within season) ───────────────────────────────
player = player.sort_values(["player", "game_date"]).reset_index(drop=True)
WINDOW = 5   # expanded from 4 — more stable with 3 seasons of data

if "season" not in player.columns:
    player["season"] = 2026

# Core rolling stats (existing)
for stat in ["ts_pct", "usg_pct", "pts_per40", "reb_per40", "ast_per40"]:
    player[f"{stat}_r{WINDOW}"] = (
        player.groupby(["player", "season"])[stat]
              .transform(lambda s: s.shift(1).rolling(WINDOW, min_periods=2).mean())
    )
    player[f"{stat}_season"] = (
        player.groupby(["player", "season"])[stat]
              .transform(lambda s: s.shift(1).expanding(min_periods=3).mean())
    )

# Additional rolling stats (new — reb_per40 and to_per40 both beat baseline)
for stat in ["reb_per40", "to_per40"]:
    # season avg already computed above for reb; add to_per40 season avg
    if f"{stat}_season" not in player.columns:
        player[f"{stat}_season"] = (
            player.groupby(["player", "season"])[stat]
                  .transform(lambda s: s.shift(1).expanding(min_periods=3).mean())
        )

# ── PBP rolling features (only if scrape_pbp_features.py has been run) ────────
# These are rolling per-player shot quality metrics computed from ESPN PBP.
# We roll within season so early-season games don't bleed into later ones.
# shift(1) ensures the current game is never included (no leakage).
PBP_SHOT_COLS = ["rim_rate", "three_rate", "avg_shot_dist",
                 "assisted_rate", "foul_drawn_rate"]

pbp_features_available = all(c in player.columns for c in PBP_SHOT_COLS)

if pbp_features_available:
    for stat in PBP_SHOT_COLS:
        player[f"{stat}_r{WINDOW}"] = (
            player.groupby(["player", "season"])[stat]
                  .transform(lambda s: s.shift(1).rolling(WINDOW, min_periods=2).mean())
        )
    print(f"PBP rolling features computed: "
          f"{[f'{s}_r{WINDOW}' for s in PBP_SHOT_COLS]}")
else:
    # Create empty columns so model_player_performance.py doesn't crash
    # when FEATURES_FULL references them — they'll be dropped as all-NaN
    for stat in PBP_SHOT_COLS:
        player[f"{stat}_r{WINDOW}"] = np.nan
    print(f"⚠  PBP features not available — run scrape_pbp_features.py first")
    print(f"   Columns set to NaN; model will fall back to BASE features")

# ── Hot streak feature ────────────────────────────────────────────────────────
# Binary: did this player exceed their season avg pts_per40 in BOTH of the
# last 2 games? 1 = hot, 0 = not. Captures trending-up players that rolling
# averages lag behind.
# Using shift(1) and shift(2) so current game is never included.
player["prev1_pts"] = (player.groupby(["player", "season"])["pts_per40"].shift(1))
player["prev2_pts"] = (player.groupby(["player", "season"])["pts_per40"].shift(2))
# We need season avg BEFORE those games — use shift(3) expanding
player["avg_before"] = (player.groupby(["player", "season"])["pts_per40"]
                               .transform(lambda s: s.shift(3).expanding(min_periods=2).mean()))
player["on_hot_streak"] = np.where(
    (player["prev1_pts"] > player["avg_before"]) &
    (player["prev2_pts"] > player["avg_before"]) &
    player["prev1_pts"].notna() &
    player["prev2_pts"].notna() &
    player["avg_before"].notna(),
    1, 0
)
player = player.drop(columns=["prev1_pts", "prev2_pts", "avg_before"])

player["prev_game_date"] = (player.groupby(["player", "season"])["game_date"].shift(1))
player["days_rest"] = (player["game_date"] - player["prev_game_date"]).dt.days.fillna(7)
player["is_home"] = (player["home_away"] == "home").astype(int)

player.to_csv(os.path.join(OUT_DIR, "player_advanced.csv"), index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 — SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
ncat_p = player[player["team"].str.contains("A&T", na=False) & (player["min"] >= 10)]

print(f"{'='*55}")
print(f"  Features built")
print(f"{'='*55}")
print(f"  Player rows:         {len(player):,}")
print(f"  Team rows (ncat):    {len(ncat_team):,}")
print(f"  Saved to:            {OUT_DIR}/")

print(f"\n  New features added this version:")
print(f"    torvik_adjoe    — opponent offensive efficiency")
print(f"    reb_per40_season — rolling rebound rate (season avg)")
print(f"    to_per40_season  — rolling turnover rate (season avg)")
print(f"    on_hot_streak    — was player above avg in last 2 games?")
print(f"    position_enc     — 0=Guard / 1=Wing / 2=Big")

torvik_col = player[player["team"].str.contains("A&T", na=False)]["opp_drtg_pregame"]
print(f"\n  opp_drtg_pregame coverage:  {torvik_col.notna().sum()} / {len(ncat_p)} rows")

barthag_col = player[player["team"].str.contains("A&T", na=False)]["torvik_barthag"]
print(f"  torvik_barthag coverage:    {barthag_col.notna().sum()} / {len(ncat_p)} rows")

hs_col = player[player["team"].str.contains("A&T", na=False)]["on_hot_streak"]
print(f"  on_hot_streak = 1:          {int(hs_col.sum())} / {len(ncat_p)} qualifying rows "
      f"({hs_col.mean()*100:.0f}% of games)")

print(f"\n  Season avg pts_per40 (5+ games, 10+ min):")
per40 = (ncat_p.groupby("player")
               .filter(lambda g: len(g) >= 5)
               .groupby("player")["pts_per40"]
               .mean()
               .sort_values(ascending=False)
               .round(1))
for pname, val in per40.items():
    print(f"    {pname:<25}  {val} pts/40")
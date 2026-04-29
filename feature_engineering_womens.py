"""
feature_engineering_womens.py
------------------------------
Identical pipeline to feature_engineering.py but for NC A&T Women's Basketball.

Data sources:
  games_raw_womens.parquet        - NC A&T women's 3-season box scores
  ncaa_box_scores_womens.parquet  - D1 women's games for opponent context
  assets/data/torvik_womens_XXXX.csv - Bart Torvik women's efficiency ratings

Run:
    python3 feature_engineering_womens.py

Before running, download Torvik women's files:
    curl "https://barttorvik.com/2024_team_results_women.csv" -o assets/data/torvik_womens_2024.csv
    curl "https://barttorvik.com/2025_team_results_women.csv" -o assets/data/torvik_womens_2025.csv
    curl "https://barttorvik.com/2026_team_results_women.csv" -o assets/data/torvik_womens_2026.csv

    Note: If those URLs 404, try:
    https://barttorvik.com/trankw.php (browse Torvik women's manually to find CSV links)
"""

import pandas as pd
import numpy as np
import os
from difflib import get_close_matches

OUT_DIR      = "assets/data"
NCAT_PATH    = os.path.join(OUT_DIR, "games_raw_womens.parquet")
D1_PATH      = os.path.join(OUT_DIR, "ncaa_box_scores_womens.parquet")

os.makedirs(OUT_DIR, exist_ok=True)

for p in [NCAT_PATH]:
    if not os.path.exists(p):
        raise SystemExit(
            f"\n{p} not found.\n"
            f"Run: python3 update_boxscores_womens.py --multi_season\n"
        )

if not os.path.exists(D1_PATH):
    print(f"⚠  {D1_PATH} not found — opponent rolling ratings unavailable.")
    print(f"   Run: caffeinate -i python3 update_boxscores_womens.py --all_d1 --season 2026")
    print(f"   Continuing with Torvik-only opponent ratings.\n")
    D1_AVAILABLE = False
else:
    D1_AVAILABLE = True


# ═══════════════════════════════════════════════════════════════════════════════
# PART 0 — LOAD TORVIK WOMEN'S RATINGS
# ═══════════════════════════════════════════════════════════════════════════════

MANUAL_CORRECTIONS_W = {
    # Add women's-specific name corrections here as needed
    # Same structure as men's MANUAL_CORRECTIONS
    "south carolina state bulldogs" : "South Carolina St.",
    "morgan state bears"            : "Morgan St.",
    "unc greensboro spartans"       : "UNC Greensboro",
    "unc wilmington seahawks"       : "UNC Wilmington",
    "william & mary tribe"          : "William & Mary",
    "campbell fighting camels"      : "Campbell",
    "stony brook seawolves"         : "Stony Brook",
    "north carolina central eagles" : "North Carolina Central",
    "maryland eastern shore hawks"  : "Maryland Eastern Shore",
}

def make_espn_to_torvik_w(torvik_names):
    def espn_to_torvik(espn_name):
        if pd.isna(espn_name):
            return None
        key = espn_name.strip().lower()
        if key in MANUAL_CORRECTIONS_W:
            return MANUAL_CORRECTIONS_W[key]
        words = espn_name.strip().split()
        for n in [len(words), len(words)-1, len(words)-2]:
            if n < 1: break
            candidate = " ".join(words[:n])
            matches = get_close_matches(candidate, torvik_names, n=1, cutoff=0.65)
            if matches:
                return matches[0]
        return None
    return espn_to_torvik


torvik_by_season = {}
for season_yr in [2024, 2025, 2026]:
    path = os.path.join(OUT_DIR, f"torvik_womens_{season_yr}.csv")
    if os.path.exists(path):
        df_t = pd.read_csv(path)
        # Torvik women's CSV may use different column names — handle both
        col_map = {}
        cols_lower = {c.lower(): c for c in df_t.columns}
        for want, alts in [
            ("adjde", ["adjde", "adj_de", "adj de"]),
            ("adjoe", ["adjoe", "adj_oe", "adj oe"]),
            ("adjt",  ["adjt",  "adj_t",  "adj t", "tempo"]),
            ("barthag", ["barthag", "bart_hag"]),
            ("team", ["team", "school", "name"]),
        ]:
            for alt in alts:
                if alt in cols_lower:
                    col_map[cols_lower[alt]] = want
                    break
        df_t = df_t.rename(columns=col_map)
        needed = ["team", "adjde", "adjoe", "adjt", "barthag"]
        present = [c for c in needed if c in df_t.columns]
        if "team" in present and "adjde" in present:
            lookup = df_t.set_index("team")[[c for c in present if c != "team"]].to_dict("index")
            torvik_by_season[season_yr] = lookup
            print(f"Torvik women's {season_yr}: {len(lookup)} teams loaded")
        else:
            print(f"⚠  torvik_womens_{season_yr}.csv missing expected columns. "
                  f"Found: {list(df_t.columns)[:8]}")
    else:
        print(f"⚠  torvik_womens_{season_yr}.csv not found")

if not torvik_by_season:
    print("\n⚠  No Torvik women's files found. Download them:")
    print("   curl https://barttorvik.com/2024_team_results_women.csv "
          "-o assets/data/torvik_womens_2024.csv")
    print("   curl https://barttorvik.com/2025_team_results_women.csv "
          "-o assets/data/torvik_womens_2025.csv")
    print("   curl https://barttorvik.com/2026_team_results_women.csv "
          "-o assets/data/torvik_womens_2026.csv\n")
print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — OPPONENT ROLLING RATINGS FROM D1 WOMEN'S DATA
# ═══════════════════════════════════════════════════════════════════════════════

STAT_COLS = ["min", "fgm", "fga", "fg2m", "fg2a", "fg3m", "fg3a",
             "ftm", "fta", "pts", "reb", "ast", "to", "stl", "blk",
             "oreb", "dreb", "pf"]

safe = lambda n, d: np.where(d > 0, n / d, 0.0)

def poss(fga, oreb, to, fta):
    return fga - oreb + to + 0.44 * fta

opp_lookup = None

if D1_AVAILABLE:
    print("Loading D1 women's box scores...")
    d1 = pd.read_parquet(D1_PATH)
    d1["game_date"] = pd.to_datetime(d1["game_date"], errors="coerce")
    d1 = d1[d1["player"] != "TEAM"].copy()
    d1 = d1.rename(columns={"3pm": "fg3m", "3pa": "fg3a",
                              "2pm": "fg2m", "2pa": "fg2a"})

    print(f"  D1W rows:  {len(d1):,}")
    print(f"  D1W games: {d1['game_id'].nunique()}\n")

    opp_rename = {c: f"opp_{c}" for c in STAT_COLS}
    d1_team = d1.groupby(["game_id", "game_date", "team"], as_index=False)[STAT_COLS].sum()
    opp_side = (d1_team[["game_id", "team"] + STAT_COLS]
                .rename(columns={"team": "opp_team", **opp_rename}))
    d1_team  = d1_team.merge(opp_side, on="game_id", how="left")
    d1_team  = d1_team[d1_team["team"] != d1_team["opp_team"]].reset_index(drop=True)

    d1_team["poss"]     = poss(d1_team["fga"],     d1_team["oreb"],  d1_team["to"],  d1_team["fta"])
    d1_team["opp_poss"] = poss(d1_team["opp_fga"], d1_team["opp_oreb"], d1_team["opp_to"], d1_team["opp_fta"])
    d1_team["pace"]     = (d1_team["poss"] + d1_team["opp_poss"]) / 2
    d1_team["ortg"]     = 100 * safe(d1_team["pts"],     d1_team["pace"])
    d1_team["drtg"]     = 100 * safe(d1_team["opp_pts"], d1_team["pace"])

    d1_team = d1_team.sort_values(["team", "game_date"]).reset_index(drop=True)
    for metric in ["ortg", "drtg", "pace"]:
        d1_team[f"{metric}_r5"] = (
            d1_team.groupby("team")[metric]
                   .transform(lambda s: s.shift(1).rolling(5, min_periods=2).mean())
        )

    if "season" in d1.columns and "season" not in d1_team.columns:
        d1_team = d1_team.merge(
            d1.drop_duplicates("game_id")[["game_id", "season"]], on="game_id", how="left"
        )

    opp_lookup = (d1_team[["game_id", "team", "drtg_r5", "ortg_r5", "pace_r5"]]
                  .rename(columns={"team": "opp_team_name",
                                   "drtg_r5": "opp_drtg_pregame",
                                   "ortg_r5": "opp_ortg_pregame",
                                   "pace_r5": "opp_pace_pregame"}))
else:
    print("Skipping D1 rolling ratings — using Torvik only.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — NC A&T WOMEN'S PLAYER FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

print("Loading NC A&T women's box scores...")
ncat = pd.read_parquet(NCAT_PATH)
ncat["game_date"] = pd.to_datetime(ncat["game_date"], errors="coerce")
ncat = ncat[ncat["player"] != "TEAM"].copy()
ncat = ncat.rename(columns={"3pm": "fg3m", "3pa": "fg3a",
                              "2pm": "fg2m", "2pa": "fg2a"})

print(f"  NC A&T W rows:  {len(ncat):,}")
print(f"  NC A&T W games: {ncat['game_id'].nunique()}\n")

# Filter to NC A&T team rows only
ncat_team_rows = ncat[ncat["team"].str.contains("A&T", na=False)].copy()

ncat_team = (ncat.groupby(["game_id", "game_date", "team"], as_index=False)[STAT_COLS].sum())
ncat_meta = (ncat[["game_id", "ncat_score", "opp_score", "home_away", "win", "opponent"]]
             .drop_duplicates("game_id"))
ncat_team = ncat_team.merge(ncat_meta, on="game_id", how="left")
ncat_team["is_ncat"]    = ncat_team["team"].str.contains("A&T", na=False).astype(int)
ncat_team["team_score"] = np.where(ncat_team["is_ncat"] == 1,
                                    ncat_team["ncat_score"], ncat_team["opp_score"])
ncat_team["opp_score_"] = np.where(ncat_team["is_ncat"] == 1,
                                    ncat_team["opp_score"], ncat_team["ncat_score"])

opp_rename = {c: f"opp_{c}" for c in STAT_COLS}
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
ncat_team["ts_pct"]   = safe(ncat_team["team_score"],
                             2 * (ncat_team["fga"] + 0.44 * ncat_team["fta"]))

ncat_team = ncat_team.sort_values(["team", "game_date"]).reset_index(drop=True)
ncat_team.to_csv(os.path.join(OUT_DIR, "team_advanced_womens.csv"), index=False)

# ── Opponent ratings ──────────────────────────────────────────────────────────
opp_names = (ncat_team[ncat_team["is_ncat"] == 0][["game_id", "team"]]
             .rename(columns={"team": "opp_team_name"}))

if opp_lookup is not None:
    opp_ratings = opp_names.merge(opp_lookup, on=["game_id", "opp_team_name"], how="left")
else:
    opp_ratings = opp_names.copy()
    opp_ratings["opp_drtg_pregame"] = np.nan
    opp_ratings["opp_ortg_pregame"] = np.nan
    opp_ratings["opp_pace_pregame"] = np.nan

# Torvik overlay
if torvik_by_season:
    game_meta = ncat_meta[["game_id", "opponent"]].copy()
    if "season" in ncat.columns:
        game_meta = game_meta.merge(
            ncat.drop_duplicates("game_id")[["game_id", "season"]], on="game_id", how="left"
        )
    else:
        game_meta["season"] = 2026

    def get_torvik_metric_w(row, metric):
        season = int(row.get("season", 2026))
        lookup = torvik_by_season.get(season, torvik_by_season.get(2026, {}))
        if not lookup: return np.nan
        mapper = make_espn_to_torvik_w(list(lookup.keys()))
        tname  = mapper(row["opponent"])
        if tname and tname in lookup:
            return lookup[tname].get(metric, np.nan)
        return np.nan

    for metric in ["adjde", "adjoe", "adjt", "barthag"]:
        game_meta[f"torvik_{metric}"] = game_meta.apply(
            lambda row: get_torvik_metric_w(row, metric), axis=1
        )

    opp_ratings = opp_ratings.merge(
        game_meta[["game_id", "torvik_adjde", "torvik_adjoe",
                   "torvik_adjt", "torvik_barthag"]],
        on="game_id", how="left"
    )
    opp_ratings["opp_drtg_pregame"] = opp_ratings["torvik_adjde"].combine_first(
        opp_ratings["opp_drtg_pregame"])
    opp_ratings["opp_ortg_pregame"] = opp_ratings["torvik_adjoe"].combine_first(
        opp_ratings["opp_ortg_pregame"])
    opp_ratings["opp_pace_pregame"] = opp_ratings["torvik_adjt"].combine_first(
        opp_ratings["opp_pace_pregame"])
    for col in ["torvik_barthag", "torvik_adjoe"]:
        m = opp_ratings[col].median()
        if not pd.isna(m):
            opp_ratings[col] = opp_ratings[col].fillna(m)
else:
    opp_ratings["torvik_barthag"] = np.nan
    opp_ratings["torvik_adjoe"]   = np.nan

coverage = opp_ratings["opp_drtg_pregame"].notna().sum()
print(f"Opponent difficulty coverage: "
      f"{coverage}/{len(opp_ratings)} games "
      f"({coverage/max(len(opp_ratings),1)*100:.0f}%)\n")

# ── Player advanced metrics ───────────────────────────────────────────────────
player = ncat.copy()

player = player.merge(
    ncat_team[["game_id", "team", "min", "fga", "fta", "to", "poss",
               "ortg", "drtg", "pace"]
              ].rename(columns={"min": "team_min", "fga": "team_fga",
                                 "fta": "team_fta", "to": "team_to",
                                 "poss": "team_poss"}),
    on=["game_id", "team"], how="left"
)

player = player.merge(
    opp_ratings[["game_id", "opp_drtg_pregame", "opp_ortg_pregame",
                 "opp_pace_pregame", "torvik_barthag", "torvik_adjoe"]],
    on="game_id", how="left"
)

# PBP shot quality features (if available)
PBP_PATH = os.path.join(OUT_DIR, "pbp_features_womens.csv")
if os.path.exists(PBP_PATH):
    pbp = pd.read_csv(PBP_PATH)
    pbp["game_id"]    = pbp["game_id"].astype(str)
    pbp["athlete_id"] = pbp["athlete_id"].astype(str)
    pbp_cols = ["game_id", "athlete_id", "rim_rate", "three_rate", "mid_rate",
                "avg_shot_dist", "assisted_rate", "foul_drawn_rate"]
    pbp = pbp[[c for c in pbp_cols if c in pbp.columns]].copy()

    # Build player→athlete_id lookup from the raw parquet (CBBpy stores ESPN IDs)
    pbp_merged = False
    if "player_id" in ncat.columns and ncat["player_id"].notna().any():
        id_map = (ncat[["player", "game_id", "player_id"]]
                  .dropna(subset=["player_id"])
                  .drop_duplicates()
                  .assign(game_id=lambda d: d["game_id"].astype(str),
                          player_id=lambda d: d["player_id"].astype(str)))

        if len(id_map) > 0:
            player["game_id"] = player["game_id"].astype(str)
            player = player.merge(id_map, on=["player", "game_id"], how="left")
            # Only merge PBP if player_id column was actually added
            if "player_id" in player.columns and player["player_id"].notna().any():
                player = player.merge(pbp, left_on=["game_id", "player_id"],
                                      right_on=["game_id", "athlete_id"], how="left")
                player = player.drop(columns=["athlete_id", "player_id"], errors="ignore")
                matched = player["rim_rate"].notna().sum()
                print(f"PBP features merged (women's): {matched} / {len(player)} rows matched")
                pbp_merged = True
            else:
                player = player.drop(columns=["player_id"], errors="ignore")
                print("  ⚠  player_id lookup produced no matches")

    if not pbp_merged:
        for col in ["rim_rate", "three_rate", "mid_rate",
                    "avg_shot_dist", "assisted_rate", "foul_drawn_rate"]:
            player[col] = np.nan
        print("  ⚠  PBP features set to NaN — excluded from model (prevents noise imputation)")
else:
    for col in ["rim_rate", "three_rate", "mid_rate",
                "avg_shot_dist", "assisted_rate", "foul_drawn_rate"]:
        player[col] = np.nan
    print("⚠  pbp_features_womens.csv not found — run scrape_pbp_features_womens.py")

# True Shooting %
player["ts_pct"]  = safe(player["pts"], 2 * (player["fga"] + 0.44 * player["fta"]))
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

# Position encoding
POSITION_MAP = {
    "G": 0, "PG": 0, "SG": 0, "G/F": 0,
    "F": 1, "SF": 1, "PF": 1, "F/G": 1, "F/C": 1,
    "C": 2,
}
if "position" in player.columns:
    player["position_enc"] = (player["position"].str.strip().str.upper()
                               .map(POSITION_MAP).fillna(1))
else:
    player["position_enc"] = 1

# Rolling features
player = player.sort_values(["player", "game_date"]).reset_index(drop=True)
WINDOW = 5

if "season" not in player.columns:
    player["season"] = 2026

for stat in ["ts_pct", "usg_pct", "pts_per40", "reb_per40", "ast_per40"]:
    player[f"{stat}_r{WINDOW}"] = (
        player.groupby(["player", "season"])[stat]
              .transform(lambda s: s.shift(1).rolling(WINDOW, min_periods=2).mean())
    )
    player[f"{stat}_season"] = (
        player.groupby(["player", "season"])[stat]
              .transform(lambda s: s.shift(1).expanding(min_periods=3).mean())
    )

for stat in ["reb_per40", "to_per40"]:
    if f"{stat}_season" not in player.columns:
        player[f"{stat}_season"] = (
            player.groupby(["player", "season"])[stat]
                  .transform(lambda s: s.shift(1).expanding(min_periods=3).mean())
        )

# PBP rolling features
PBP_SHOT_COLS = ["rim_rate", "three_rate", "avg_shot_dist",
                 "assisted_rate", "foul_drawn_rate"]
for stat in PBP_SHOT_COLS:
    if stat in player.columns and player[stat].notna().any():
        player[f"{stat}_r{WINDOW}"] = (
            player.groupby(["player", "season"])[stat]
                  .transform(lambda s: s.shift(1).rolling(WINDOW, min_periods=2).mean())
        )
    else:
        player[f"{stat}_r{WINDOW}"] = np.nan

# Hot streak
player["prev1_pts"] = player.groupby(["player", "season"])["pts_per40"].shift(1)
player["prev2_pts"] = player.groupby(["player", "season"])["pts_per40"].shift(2)
player["avg_before"] = (player.groupby(["player", "season"])["pts_per40"]
                               .transform(lambda s: s.shift(3).expanding(min_periods=2).mean()))
player["on_hot_streak"] = np.where(
    (player["prev1_pts"] > player["avg_before"]) &
    (player["prev2_pts"] > player["avg_before"]) &
    player["prev1_pts"].notna() & player["prev2_pts"].notna() &
    player["avg_before"].notna(), 1, 0
)
player = player.drop(columns=["prev1_pts", "prev2_pts", "avg_before"], errors="ignore")

player["prev_game_date"] = player.groupby(["player", "season"])["game_date"].shift(1)
player["days_rest"] = (player["game_date"] - player["prev_game_date"]).dt.days.fillna(7)
player["is_home"]   = (player["home_away"] == "home").astype(int)

player.to_csv(os.path.join(OUT_DIR, "player_advanced_womens.csv"), index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
ncat_p = player[player["team"].str.contains("A&T", na=False) & (player["min"] >= 10)]

print(f"\n{'='*55}")
print(f"  Women's features built")
print(f"{'='*55}")
print(f"  Player rows:      {len(player):,}")
print(f"  Qualifying rows:  {len(ncat_p):,} (10+ min)")
print(f"  Saved to:         {OUT_DIR}/player_advanced_womens.csv")

print(f"\n  Season avg pts_per40 (5+ games, 10+ min):")
per40 = (ncat_p.groupby("player")
               .filter(lambda g: len(g) >= 5)
               .groupby("player")["pts_per40"]
               .mean()
               .sort_values(ascending=False)
               .round(1))
for pname, val in per40.items():
    print(f"    {pname:<25}  {val} pts/40")
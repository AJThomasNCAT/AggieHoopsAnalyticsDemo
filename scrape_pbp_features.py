"""
scrape_pbp_features.py
----------------------
Scrapes ESPN's play-by-play summary API for all NC A&T games and computes
shot quality features that box scores can't provide.

Why this improves the model:
  Box scores tell us Walker scored 22 pts/40 — but not HOW he scored.
  A player getting 60% of shots at the rim is more reliably productive
  than one shooting 40% from mid-range. Shot quality metrics capture
  this dimension that TS% and eFG% partially miss.

Features computed per player per game:
  rim_rate         — % of shots attempted within ~8ft of basket (high = efficient scorer)
  three_rate       — % of shots that were 3-point attempts
  mid_rate         — % of shots that were mid-range (the "bad" shot in modern analytics)
  avg_shot_dist    — average shot distance (lower = more rim-seeking)
  assisted_rate    — % of made shots that were assisted (depends on playmaking)
  foul_drawn_rate  — free throw attempts per shot attempt (floor-raiser)
  pts_in_paint     — points scored in the paint (direct per-game, not box-score estimate)

Opponent-level features (for use as pregame context):
  opp_rim_rate_allowed    — how often does this opponent give up rim attempts
  opp_three_rate_allowed  — does this defense force 3s or allow them
  opp_avg_shot_dist_allowed — how deep does this defense push shooters

API endpoint used (no key required):
  https://site.api.espn.com/apis/site/v2/sports/basketball/
    mens-college-basketball/summary?event={game_id}

Run:
    python3 scrape_pbp_features.py

Output:
    assets/data/pbp_features.csv      — one row per player per game
    assets/data/pbp_game_features.csv — one row per team per game (opponent context)
"""

import requests
import pandas as pd
import numpy as np
import os
import time
import json
from math import sqrt

OUT_DIR    = "assets/data"
RAW_PATH   = os.path.join(OUT_DIR, "games_raw.parquet")
OUT_PBP    = os.path.join(OUT_DIR, "pbp_features.csv")
OUT_GAME   = os.path.join(OUT_DIR, "pbp_game_features.csv")

os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(RAW_PATH):
    raise SystemExit(f"\n{RAW_PATH} not found. Run update_boxscores.py first.\n")

# ── Load game IDs ─────────────────────────────────────────────────────────────
raw = pd.read_parquet(RAW_PATH)
raw["game_date"] = pd.to_datetime(raw["game_date"], errors="coerce")

# One row per game (game_id is unique per game)
games = (raw[raw["team"].str.contains("A&T", na=False)]
         .drop_duplicates("game_id")[["game_id", "game_date", "opponent", "season",
                                       "home_away", "win"]]
         .sort_values("game_date")
         .reset_index(drop=True))

print(f"Games to scrape: {len(games)}")
print(f"Date range: {games['game_date'].min().date()} → {games['game_date'].max().date()}\n")


# ── Shot distance helper ──────────────────────────────────────────────────────
# ESPN coordinates: basket is at roughly (25, 5.25) in their coordinate system
# (half-court view, measured in feet, origin at corner).
# We compute Euclidean distance from basket.
BASKET_X = 25.0   # center of court (width = 50ft)
BASKET_Y = 5.25   # ~5.25ft from baseline to center of basket

def shot_distance(x, y):
    """Approximate distance from basket in feet."""
    if x is None or y is None:
        return None
    try:
        return sqrt((float(x) - BASKET_X)**2 + (float(y) - BASKET_Y)**2)
    except (TypeError, ValueError):
        return None

def classify_zone(dist):
    """Classify shot distance into zone."""
    if dist is None:
        return "unknown"
    if dist <= 8:
        return "rim"        # layup/dunk territory
    elif dist <= 16:
        return "midrange"   # the analytically "bad" shot
    else:
        return "three"      # perimeter / 3-point territory

def shot_type_from_text(text):
    """Extract shot type from play description text."""
    if not text:
        return "unknown"
    text_lower = text.lower()
    if any(w in text_lower for w in ["layup", "dunk", "tip", "alley"]):
        return "rim"
    if "three point" in text_lower or "3-point" in text_lower:
        return "three"
    if "jumper" in text_lower or "jump shot" in text_lower:
        return "jumper"
    if "hook" in text_lower:
        return "hook"
    return "other"


# ── ESPN API fetcher ──────────────────────────────────────────────────────────
def fetch_espn_summary(game_id):
    """Fetch the ESPN summary API for a game. Returns the JSON dict or None."""
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/basketball/"
        f"mens-college-basketball/summary?event={game_id}"
    )
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"    ⚠  Failed to fetch {game_id}: {e}")
        return None


# ── Main scrape loop ──────────────────────────────────────────────────────────
player_rows = []   # one dict per player per game
game_rows   = []   # one dict per team per game (for opponent context)
failed      = []

for idx, game in games.iterrows():
    gid      = game["game_id"]
    opp_name = game["opponent"]
    g_date   = game["game_date"]
    season   = game.get("season", 2026)

    print(f"  [{idx+1:>2}/{len(games)}] {g_date.strftime('%Y-%m-%d')}  "
          f"NC A&T vs {opp_name}")

    data = fetch_espn_summary(gid)
    if data is None or "plays" not in data:
        print(f"    → No PBP data available")
        failed.append(gid)
        time.sleep(0.5)
        continue

    plays = data["plays"]
    shooting_plays = [p for p in plays if p.get("shootingPlay", False)]

    if not shooting_plays:
        print(f"    → No shooting plays found")
        failed.append(gid)
        time.sleep(0.5)
        continue

    # Identify NC A&T's ESPN team ID from the header
    ncat_team_id = None
    try:
        competitors = data["header"]["competitions"][0]["competitors"]
        for comp in competitors:
            if "a&t" in comp["team"]["displayName"].lower() or \
               "north carolina a" in comp["team"]["displayName"].lower():
                ncat_team_id = str(comp["team"]["id"])
                break
    except (KeyError, IndexError):
        pass

    # ── Per-player shot stats ─────────────────────────────────────────────────
    # Group shooting plays by athlete ID
    player_shots = {}   # athlete_id → list of play dicts

    for play in shooting_plays:
        participants = play.get("participants", [])
        if not participants:
            continue
        athlete = participants[0].get("athlete", {})
        athlete_id = str(athlete.get("id", ""))
        if not athlete_id:
            continue

        team_id = str(play.get("team", {}).get("id", ""))
        is_ncat = (team_id == ncat_team_id) if ncat_team_id else True

        # Get shot coordinates
        coord = play.get("coordinate", {})
        x = coord.get("x")
        y = coord.get("y")
        dist = shot_distance(x, y)

        play_info = {
            "athlete_id": athlete_id,
            "team_id":    team_id,
            "is_ncat":    is_ncat,
            "made":       play.get("scoringPlay", False),
            "score_value": play.get("scoreValue", 0),
            "text":       play.get("text", ""),
            "x":          x,
            "y":          y,
            "dist":       dist,
            "zone":       classify_zone(dist) if dist else shot_type_from_text(play.get("text", "")),
            "is_three":   (play.get("scoreValue", 0) == 3 or
                           "three point" in play.get("text", "").lower()),
            "assisted":   "assisted" in play.get("text", "").lower(),
        }
        player_shots.setdefault(athlete_id, []).append(play_info)

    # ── Foul draws (free throw attempts per player) ───────────────────────────
    # Look for free throw plays to count FTA per player in this game
    ft_plays = [p for p in plays
                if p.get("type", {}).get("text", "").lower() in
                   ["free throw", "free throw - 1 of 2", "free throw - 2 of 2",
                    "free throw - 1 of 3", "free throw - 2 of 3", "free throw - 3 of 3"]]
    ft_attempts = {}
    for ft in ft_plays:
        participants = ft.get("participants", [])
        if participants:
            aid = str(participants[0].get("athlete", {}).get("id", ""))
            if aid:
                ft_attempts[aid] = ft_attempts.get(aid, 0) + 1

    # ── Win probability context ───────────────────────────────────────────────
    # Average win probability for NC A&T across the game
    wp_data = data.get("winprobability", [])
    ncat_avg_wp = None
    if wp_data and ncat_team_id:
        wps = []
        for wp in wp_data:
            team_id_wp = str(wp.get("teamId", ""))
            if team_id_wp == ncat_team_id:
                wps.append(wp.get("probability", 0.5))
        if wps:
            ncat_avg_wp = round(np.mean(wps), 3)

    # ── Build player rows ─────────────────────────────────────────────────────
    for athlete_id, shot_list in player_shots.items():
        total_shots = len(shot_list)
        if total_shots == 0:
            continue

        made_shots = [s for s in shot_list if s["made"]]
        rim_shots  = [s for s in shot_list if s["zone"] == "rim"]
        mid_shots  = [s for s in shot_list if s["zone"] == "midrange"]
        three_shots = [s for s in shot_list if s["is_three"]]
        assisted_makes = [s for s in made_shots if s["assisted"]]

        dists = [s["dist"] for s in shot_list if s["dist"] is not None]

        player_rows.append({
            "game_id":        gid,
            "game_date":      g_date,
            "season":         season,
            "athlete_id":     athlete_id,
            "is_ncat":        shot_list[0]["is_ncat"],
            "opponent":       opp_name,
            # Shot volume
            "total_shots":    total_shots,
            "made_shots":     len(made_shots),
            # Shot quality features
            "rim_rate":       len(rim_shots) / total_shots,
            "mid_rate":       len(mid_shots) / total_shots,
            "three_rate":     len(three_shots) / total_shots,
            "avg_shot_dist":  round(np.mean(dists), 2) if dists else None,
            # Playmaking dependence
            "assisted_rate":  len(assisted_makes) / max(len(made_shots), 1),
            # Foul drawing
            "fta_this_game":  ft_attempts.get(athlete_id, 0),
            "foul_drawn_rate": ft_attempts.get(athlete_id, 0) / total_shots,
            # Game context
            "ncat_avg_wp":    ncat_avg_wp,
        })

    # ── Team-level shot stats for opponent context ────────────────────────────
    for team_label in ["ncat", "opp"]:
        is_ncat_flag = (team_label == "ncat")
        team_shots = [s for shots in player_shots.values()
                       for s in shots if s["is_ncat"] == is_ncat_flag]

        if not team_shots:
            continue

        total = len(team_shots)
        rim   = [s for s in team_shots if s["zone"] == "rim"]
        mid   = [s for s in team_shots if s["zone"] == "midrange"]
        three = [s for s in team_shots if s["is_three"]]
        dists = [s["dist"] for s in team_shots if s["dist"] is not None]

        game_rows.append({
            "game_id":     gid,
            "game_date":   g_date,
            "season":      season,
            "team_label":  team_label,
            "opponent":    opp_name,
            "total_shots": total,
            "rim_rate":    len(rim) / total,
            "mid_rate":    len(mid) / total,
            "three_rate":  len(three) / total,
            "avg_shot_dist": round(np.mean(dists), 2) if dists else None,
        })

    print(f"    ✓  {len([r for r in player_rows if r['game_id']==gid and r['is_ncat']])} "
          f"NC A&T players | {len(shooting_plays)} total shot plays")

    time.sleep(0.7)   # polite pause


# ── Save outputs ──────────────────────────────────────────────────────────────
player_df = pd.DataFrame(player_rows)
game_df   = pd.DataFrame(game_rows)

# Filter to NC A&T players only for the player features file
ncat_player_df = player_df[player_df["is_ncat"] == True].copy()

ncat_player_df.to_csv(OUT_PBP,  index=False)
game_df.to_csv(OUT_GAME, index=False)


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  PBP scrape complete")
print(f"{'='*55}")
print(f"  Games scraped:        {len(games) - len(failed)} / {len(games)}")
print(f"  Failed games:         {len(failed)}")
print(f"  NC A&T player rows:   {len(ncat_player_df):,}")
print(f"  Team-game rows:       {len(game_df):,}")
print(f"\n  Saved:")
print(f"    {OUT_PBP}")
print(f"    {OUT_GAME}")

if len(ncat_player_df) > 0:
    print(f"\n  Feature ranges (sanity check):")
    for col in ["rim_rate", "three_rate", "avg_shot_dist", "assisted_rate"]:
        if col in ncat_player_df.columns:
            vals = ncat_player_df[col].dropna()
            print(f"    {col:<20}  {vals.min():.3f} – {vals.max():.3f}  "
                  f"(mean {vals.mean():.3f})")

    print(f"\n  Top 5 rim-seeking players (season avg):")
    top_rim = (ncat_player_df.groupby("athlete_id")["rim_rate"]
                              .mean()
                              .sort_values(ascending=False)
                              .head(5))
    for aid, rate in top_rim.items():
        print(f"    athlete_id {aid}: {rate:.1%} rim rate")

if failed:
    print(f"\n  Failed game IDs (no PBP on ESPN): {failed}")
    print(f"  These will use NaN for PBP features — imputed with median in feature_engineering.py")

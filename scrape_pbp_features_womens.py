"""
scrape_pbp_features_womens.py
------------------------------
Identical to scrape_pbp_features.py but for NC A&T Women's Basketball.
Uses the womens-college-basketball ESPN API slug.

Run:
    python3 scrape_pbp_features_womens.py

Output:
    assets/data/pbp_features_womens.csv
    assets/data/pbp_game_features_womens.csv
"""

import requests
import pandas as pd
import numpy as np
import os
import time
from math import sqrt

OUT_DIR   = "assets/data"
RAW_PATH  = os.path.join(OUT_DIR, "games_raw_womens.parquet")
OUT_PBP   = os.path.join(OUT_DIR, "pbp_features_womens.csv")
OUT_GAME  = os.path.join(OUT_DIR, "pbp_game_features_womens.csv")

SPORT_SLUG = "womens-college-basketball"   # ← only difference from men's

os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(RAW_PATH):
    raise SystemExit(
        f"\n{RAW_PATH} not found.\n"
        f"Run: python3 update_boxscores_womens.py --multi_season\n"
    )

# ── Load game IDs ─────────────────────────────────────────────────────────────
raw = pd.read_parquet(RAW_PATH)
raw["game_date"] = pd.to_datetime(raw["game_date"], errors="coerce")

games = (raw[raw["team"].str.contains("A&T", na=False)]
         .drop_duplicates("game_id")[["game_id", "game_date", "opponent",
                                       "season", "home_away", "win"]]
         .sort_values("game_date")
         .reset_index(drop=True))

print(f"Games to scrape: {len(games)}")
print(f"Date range: {games['game_date'].min().date()} → {games['game_date'].max().date()}\n")


# ── Shot helpers (identical to men's) ────────────────────────────────────────
BASKET_X = 25.0
BASKET_Y = 5.25

def shot_distance(x, y):
    if x is None or y is None:
        return None
    try:
        return sqrt((float(x) - BASKET_X)**2 + (float(y) - BASKET_Y)**2)
    except (TypeError, ValueError):
        return None

def classify_zone(dist):
    if dist is None: return "unknown"
    if dist <= 8:    return "rim"
    if dist <= 16:   return "midrange"
    return "three"

def shot_type_from_text(text):
    if not text: return "unknown"
    t = text.lower()
    if any(w in t for w in ["layup", "dunk", "tip", "alley"]): return "rim"
    if "three point" in t or "3-point" in t: return "three"
    if "jumper" in t or "jump shot" in t: return "jumper"
    return "other"


# ── ESPN API fetcher ──────────────────────────────────────────────────────────
def fetch_espn_summary(game_id):
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/basketball/"
        f"{SPORT_SLUG}/summary?event={game_id}"
    )
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"    ⚠  Failed to fetch {game_id}: {e}")
        return None


# ── Main scrape loop ──────────────────────────────────────────────────────────
player_rows = []
game_rows   = []
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

    plays          = data["plays"]
    shooting_plays = [p for p in plays if p.get("shootingPlay", False)]

    if not shooting_plays:
        print(f"    → No shooting plays found")
        failed.append(gid)
        time.sleep(0.5)
        continue

    # Identify NC A&T's ESPN team ID
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

    # Group shots by athlete
    player_shots = {}
    for play in shooting_plays:
        participants = play.get("participants", [])
        if not participants:
            continue
        athlete    = participants[0].get("athlete", {})
        athlete_id = str(athlete.get("id", ""))
        if not athlete_id:
            continue

        team_id = str(play.get("team", {}).get("id", ""))
        is_ncat = (team_id == ncat_team_id) if ncat_team_id else True

        coord = play.get("coordinate", {})
        x, y  = coord.get("x"), coord.get("y")
        dist  = shot_distance(x, y)

        play_info = {
            "athlete_id": athlete_id,
            "team_id":    team_id,
            "is_ncat":    is_ncat,
            "made":       play.get("scoringPlay", False),
            "score_value": play.get("scoreValue", 0),
            "text":       play.get("text", ""),
            "x": x, "y": y,
            "dist":     dist,
            "zone":     classify_zone(dist) if dist else shot_type_from_text(play.get("text", "")),
            "is_three": (play.get("scoreValue", 0) == 3 or
                         "three point" in play.get("text", "").lower()),
            "assisted": "assisted" in play.get("text", "").lower(),
        }
        player_shots.setdefault(athlete_id, []).append(play_info)

    # Free throw counts
    ft_plays = [p for p in plays
                if p.get("type", {}).get("text", "").lower().startswith("free throw")]
    ft_attempts = {}
    for ft in ft_plays:
        participants = ft.get("participants", [])
        if participants:
            aid = str(participants[0].get("athlete", {}).get("id", ""))
            if aid:
                ft_attempts[aid] = ft_attempts.get(aid, 0) + 1

    # Build player rows
    for athlete_id, shot_list in player_shots.items():
        total_shots = len(shot_list)
        if total_shots == 0:
            continue

        made_shots     = [s for s in shot_list if s["made"]]
        rim_shots      = [s for s in shot_list if s["zone"] == "rim"]
        mid_shots      = [s for s in shot_list if s["zone"] == "midrange"]
        three_shots    = [s for s in shot_list if s["is_three"]]
        assisted_makes = [s for s in made_shots if s["assisted"]]
        dists          = [s["dist"] for s in shot_list if s["dist"] is not None]

        player_rows.append({
            "game_id":         gid,
            "game_date":       g_date,
            "season":          season,
            "athlete_id":      athlete_id,
            "is_ncat":         shot_list[0]["is_ncat"],
            "opponent":        opp_name,
            "total_shots":     total_shots,
            "made_shots":      len(made_shots),
            "rim_rate":        len(rim_shots) / total_shots,
            "mid_rate":        len(mid_shots) / total_shots,
            "three_rate":      len(three_shots) / total_shots,
            "avg_shot_dist":   round(np.mean(dists), 2) if dists else None,
            "assisted_rate":   len(assisted_makes) / max(len(made_shots), 1),
            "fta_this_game":   ft_attempts.get(athlete_id, 0),
            "foul_drawn_rate": ft_attempts.get(athlete_id, 0) / total_shots,
        })

    # Team-level rows for opponent context
    for team_label in ["ncat", "opp"]:
        is_ncat_flag = (team_label == "ncat")
        team_shots   = [s for shots in player_shots.values()
                          for s in shots if s["is_ncat"] == is_ncat_flag]
        if not team_shots:
            continue
        total = len(team_shots)
        dists = [s["dist"] for s in team_shots if s["dist"] is not None]
        game_rows.append({
            "game_id":       gid,
            "game_date":     g_date,
            "season":        season,
            "team_label":    team_label,
            "opponent":      opp_name,
            "total_shots":   total,
            "rim_rate":      len([s for s in team_shots if s["zone"] == "rim"]) / total,
            "mid_rate":      len([s for s in team_shots if s["zone"] == "midrange"]) / total,
            "three_rate":    len([s for s in team_shots if s["is_three"]]) / total,
            "avg_shot_dist": round(np.mean(dists), 2) if dists else None,
        })

    ncat_count = len([r for r in player_rows if r["game_id"] == gid and r["is_ncat"]])
    print(f"    ✓  {ncat_count} NC A&T players | {len(shooting_plays)} total shot plays")

    time.sleep(0.7)


# ── Save ──────────────────────────────────────────────────────────────────────
player_df      = pd.DataFrame(player_rows)
game_df        = pd.DataFrame(game_rows)
ncat_player_df = player_df[player_df["is_ncat"] == True].copy()

ncat_player_df.to_csv(OUT_PBP,  index=False)
game_df.to_csv(OUT_GAME, index=False)

print(f"\n{'='*55}")
print(f"  Women's PBP scrape complete")
print(f"{'='*55}")
print(f"  Games scraped:        {len(games) - len(failed)} / {len(games)}")
print(f"  Failed games:         {len(failed)}")
print(f"  NC A&T player rows:   {len(ncat_player_df):,}")

if len(ncat_player_df) > 0:
    print(f"\n  Feature ranges:")
    for col in ["rim_rate", "three_rate", "avg_shot_dist", "assisted_rate"]:
        if col in ncat_player_df.columns:
            v = ncat_player_df[col].dropna()
            print(f"    {col:<20}  {v.min():.3f} – {v.max():.3f}  (mean {v.mean():.3f})")

    print(f"\n  Top rim-seeking players (season avg):")
    top = (ncat_player_df.groupby("athlete_id")["rim_rate"]
                         .mean().sort_values(ascending=False).head(5))
    for aid, rate in top.items():
        print(f"    athlete_id {aid}: {rate:.1%} rim rate")

if failed:
    print(f"\n  Failed: {failed}")
    print(f"  These will be median-imputed in feature_engineering_womens.py")

print(f"\n  Saved:")
print(f"    {OUT_PBP}")
print(f"    {OUT_GAME}")

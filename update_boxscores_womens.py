"""
update_boxscores_womens.py
--------------------------
Scrapes NC A&T WOMEN'S box scores using CBBpy (womens_scraper).
Mirrors update_boxscores.py exactly — only differences are:
  - uses cbbpy.womens_scraper instead of mens_scraper
  - ESPN sport slug is "womens-college-basketball"
  - NC A&T women's ESPN team ID
  - output files are suffixed _womens

Usage:
    python3 update_boxscores_womens.py                   # 2025-26 season
    python3 update_boxscores_womens.py --season 2025     # specific season
    python3 update_boxscores_womens.py --multi_season    # 2024+2025+2026
    caffeinate -i python3 update_boxscores_womens.py --all_d1 --season 2026
"""

import requests
import pandas as pd
import json
import os
import time
import argparse
import cbbpy.womens_scraper as s

# NC A&T Women's ESPN team ID
# Verify at: https://www.espn.com/womens-college-basketball/team/_/id/XXXX
NCAT_ESPN_ID = "2448"       # same institution ID — ESPN uses same ID for both
NCAT_NAME    = "North Carolina A&T"
SPORT_SLUG   = "womens-college-basketball"

parser = argparse.ArgumentParser()
parser.add_argument("--season",       type=int, default=2026)
parser.add_argument("--all_d1",       action="store_true")
parser.add_argument("--multi_season", action="store_true")
parser.add_argument("--output_dir",   type=str, default="assets/data")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)


# ── OPTION A: Full D1 women's scrape ─────────────────────────────────────────
if args.all_d1:
    import signal
    from datetime import date, timedelta

    CKPT_DIR = os.path.join(args.output_dir, f"checkpoint_d1_womens_{args.season}")
    os.makedirs(CKPT_DIR, exist_ok=True)

    already_done = {
        f.replace(".parquet", "")
        for f in os.listdir(CKPT_DIR)
        if f.endswith(".parquet")
    }
    print(f"Checkpoint directory: {CKPT_DIR}")
    print(f"Already scraped:      {len(already_done)} games\n")

    season_year = args.season
    start_date  = date(season_year - 1, 11, 1)
    end_date    = date(season_year,      4, 15)

    print(f"Scanning ESPN scoreboard for women's game IDs "
          f"({start_date} → {end_date})...")

    all_game_ids  = []
    all_game_info = {}

    current = start_date
    while current <= end_date:
        date_str = current.strftime("%Y%m%d")
        url = (f"https://site.api.espn.com/apis/site/v2/sports/basketball/"
               f"{SPORT_SLUG}/scoreboard?dates={date_str}&limit=200")
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            if r.status_code == 200:
                data = r.json()
                for event in data.get("events", []):
                    gid = event["id"]
                    if gid not in already_done:
                        all_game_ids.append(gid)
                        try:
                            comps = event["competitions"][0]["competitors"]
                            teams = [c["team"]["displayName"] for c in comps]
                            all_game_info[gid] = {"date": str(current), "teams": teams}
                        except (KeyError, IndexError):
                            all_game_info[gid] = {"date": str(current), "teams": []}
        except Exception as e:
            print(f"  Date {date_str}: {e}")
        current += timedelta(days=1)

    print(f"New games to scrape: {len(all_game_ids)}\n")

    graceful_stop = [False]
    def _handler(sig, frame):
        print("\n⚠  Interrupt received — finishing current game then stopping.")
        graceful_stop[0] = True
    signal.signal(signal.SIGINT,  _handler)
    signal.signal(signal.SIGTERM, _handler)

    scraped = 0
    failed  = 0

    for i, gid in enumerate(all_game_ids):
        if graceful_stop[0]:
            print("Stopping gracefully.")
            break
        info = all_game_info.get(gid, {})
        if (i + 1) % 50 == 0:
            teams_str = " vs ".join(info.get("teams", []))
            print(f"  [{i+1:>5}/{len(all_game_ids)}] {info.get('date','')}  {teams_str}")
        try:
            df = s.get_game_boxscore(gid)
            if df is not None and not df.empty:
                df["game_id"]   = gid
                df["game_date"] = info.get("date", "")
                df["season"]    = args.season
                df.to_parquet(os.path.join(CKPT_DIR, f"{gid}.parquet"), index=False)
                scraped += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
        time.sleep(0.35)

    # Combine all checkpoints into one parquet
    print(f"\nCombining checkpoints...")
    frames = []
    for fname in os.listdir(CKPT_DIR):
        if fname.endswith(".parquet"):
            try:
                frames.append(pd.read_parquet(os.path.join(CKPT_DIR, fname)))
            except Exception:
                pass

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        out_path = os.path.join(args.output_dir, "ncaa_box_scores_womens.parquet")
        combined.to_parquet(out_path, index=False)
        print(f"✅  ncaa_box_scores_womens.parquet  ({len(combined):,} rows, "
              f"{combined['game_id'].nunique()} games)")
    else:
        print("No data collected.")

    raise SystemExit(0)


# ── OPTION B: NC A&T multi-season or single season ───────────────────────────
def get_ncat_game_ids(season_year):
    """
    Use ESPN's team schedule API to get all NC A&T women's game IDs for a season.
    Returns list of (game_id, game_date, opponent, home_away) tuples.
    """
    url = (f"https://site.api.espn.com/apis/site/v2/sports/basketball/"
           f"{SPORT_SLUG}/teams/{NCAT_ESPN_ID}/schedule?season={season_year}")
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  Failed to fetch schedule for {season_year}: {e}")
        return []

    results = []
    for event in data.get("events", []):
        try:
            gid       = event["id"]
            game_date = event["date"][:10]
            comp      = event["competitions"][0]
            comps     = comp["competitors"]

            # Women's API: status lives inside competitions, not at event level
            comp_status = comp.get("status", {}).get("type", {})
            completed   = comp_status.get("completed", False)
            status_name = comp_status.get("name", "")
            if not completed and status_name not in ("STATUS_FINAL", "STATUS_FINAL_OT"):
                continue

            home_team = next((c for c in comps if c["homeAway"] == "home"), comps[0])
            away_team = next((c for c in comps if c["homeAway"] == "away"), comps[1])
            is_home   = (home_team["team"]["id"] == NCAT_ESPN_ID)
            opp       = away_team["team"]["displayName"] if is_home \
                        else home_team["team"]["displayName"]
            results.append((gid, game_date, opp, "home" if is_home else "away"))
        except (KeyError, IndexError, StopIteration):
            continue
    return results


seasons = [2024, 2025, 2026] if args.multi_season else [args.season]

all_frames = []

for season_year in seasons:
    print(f"\n── Season {season_year-1}-{str(season_year)[-2:]} ──────────────────────")
    games = get_ncat_game_ids(season_year)
    print(f"  Found {len(games)} completed games")

    for idx, (gid, game_date, opponent, home_away) in enumerate(games, 1):
        if idx % 10 == 0:
            print(f"  [{idx:>2}/{len(games)}] {game_date}  vs {opponent}")
        try:
            df = s.get_game_boxscore(gid)
            if df is None or df.empty:
                continue
            # Get score info from ESPN summary API
            summary_url = (
                f"https://site.api.espn.com/apis/site/v2/sports/basketball/"
                f"{SPORT_SLUG}/summary?event={gid}"
            )
            ncat_score = opp_score = win = None
            try:
                sr = requests.get(summary_url, headers={"User-Agent": "Mozilla/5.0"},
                                  timeout=10)
                sdata = sr.json()
                comps = sdata["header"]["competitions"][0]["competitors"]
                for comp in comps:
                    score = int(comp.get("score", 0))
                    if NCAT_NAME.lower() in comp["team"]["displayName"].lower():
                        ncat_score = score
                    else:
                        opp_score = score
                if ncat_score is not None and opp_score is not None:
                    win = int(ncat_score > opp_score)
            except Exception:
                pass

            df["game_id"]   = gid
            df["game_date"] = game_date
            df["season"]    = season_year
            df["opponent"]  = opponent
            df["home_away"] = home_away
            df["ncat_score"] = ncat_score
            df["opp_score"]  = opp_score
            df["win"]        = win
            all_frames.append(df)
        except Exception as e:
            print(f"    ⚠  {gid}: {e}")
        time.sleep(0.4)

if not all_frames:
    raise SystemExit("No data collected. Check team ID and season.")

combined = pd.concat(all_frames, ignore_index=True)
combined["game_date"] = pd.to_datetime(combined["game_date"], errors="coerce")
combined = combined.sort_values("game_date").reset_index(drop=True)

# Save outputs
parquet_path = os.path.join(args.output_dir, "games_raw_womens.parquet")
csv_path     = os.path.join(args.output_dir, "games_raw_womens.csv")
combined.to_parquet(parquet_path, index=False)
combined.to_csv(csv_path, index=False)

print(f"\n✅  games_raw_womens.parquet  ({len(combined):,} rows, "
      f"{combined['game_id'].nunique()} games, "
      f"{combined['player'].nunique()} players)")
print(f"✅  games_raw_womens.csv")
print(f"\nSeasons scraped: {sorted(combined['season'].unique().tolist())}")
print(f"Date range: {combined['game_date'].min().date()} → "
      f"{combined['game_date'].max().date()}")
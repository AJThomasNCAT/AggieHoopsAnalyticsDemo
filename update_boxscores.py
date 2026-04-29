"""
update_boxscores.py
-------------------
Scrapes NC A&T box scores for the current season using CBBpy.

Strategy:
  1. Hit ESPN's public schedule API to get all NC A&T game IDs
  2. Loop through each completed game and call get_game_boxscore()
  3. Save the combined results to assets/data/games_raw.parquet
     and assets/data/games_raw.csv (for the dashboard)

Usage:
    python3 update_boxscores.py                  # current season (2026)
    python3 update_boxscores.py --season 2025    # one specific season
    python3 update_boxscores.py --multi_season   # 2024 + 2025 + 2026 combined
    python3 update_boxscores.py --all_d1         # every D1 game (slow - hours)
"""

import requests
import pandas as pd
import json
import os
import time
import argparse
import cbbpy.mens_scraper as s

# NC A&T's ESPN team ID (confirmed from espn.com URL)
NCAT_ESPN_ID = "2448"
NCAT_NAME    = "North Carolina A&T"

parser = argparse.ArgumentParser()
parser.add_argument("--season",  type=int, default=2026,
                    help="Season year (2026 = 2025-26 season)")
parser.add_argument("--all_d1",  action="store_true",
                    help="Scrape every D1 game instead of just NC A&T")
parser.add_argument("--multi_season", action="store_true",
                    help="Scrape 2024+2025+2026 NC A&T seasons and combine them")
parser.add_argument("--output_dir", type=str, default="assets/data")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)


# ── OPTION A: Full D1 scrape with checkpointing ──────────────────────────────
if args.all_d1:
    import signal
    from datetime import date, timedelta

    # ── Checkpoint directory ──────────────────────────────────────────────────
    # Each completed game is saved as its own small parquet file.
    # On restart, we scan this directory and skip any game_id already saved.
    # This means a crash at game 4,000 resumes at game 4,001, not game 0.
    CKPT_DIR = os.path.join(args.output_dir, f"checkpoint_d1_{args.season}")
    os.makedirs(CKPT_DIR, exist_ok=True)

    already_done = {
        f.replace(".parquet", "")
        for f in os.listdir(CKPT_DIR)
        if f.endswith(".parquet")
    }
    print(f"Checkpoint directory: {CKPT_DIR}")
    print(f"Already scraped:      {len(already_done)} games\n")

    # ── Step 1: Collect all game IDs via ESPN scoreboard API ─────────────────
    # Hit ESPN's scoreboard endpoint for every date in the season.
    # Season runs roughly Nov 1 → Apr 15. We overshoot on both ends — empty
    # dates return quickly and don't cost scraping time.
    #
    # Example URL:
    # https://site.api.espn.com/apis/site/v2/sports/basketball/
    #   mens-college-basketball/scoreboard?dates=20251110&limit=200

    season_year = args.season  # e.g. 2026 means 2025-26 season
    start_date  = date(season_year - 1, 11, 1)   # Nov 1 of prior year
    end_date    = date(season_year,      4, 15)   # Apr 15 of season year

    print(f"Scanning ESPN scoreboard for game IDs "
          f"({start_date} → {end_date})...")

    all_game_ids  = []   # ordered list of game IDs to scrape
    game_meta_d1  = {}   # game_id → {date, home_team, away_team, ...}

    current = start_date
    while current <= end_date:
        date_str = current.strftime("%Y%m%d")
        PAGE_SIZE = 200
        offset    = 0
        day_count = 0

        while True:
            # groups=50 is ESPN's ID for NCAA Division I Men's Basketball.
            # Without this parameter the API only returns "featured" games,
            # which is why the first run only captured ~1,156 of ~5,500 games.
            url = (
                "https://site.api.espn.com/apis/site/v2/sports/basketball/"
                f"mens-college-basketball/scoreboard"
                f"?dates={date_str}&groups=50&limit={PAGE_SIZE}&offset={offset}"
            )
            try:
                r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                r.raise_for_status()
                data   = r.json()
                events = data.get("events", [])

                for event in events:
                    comp   = event["competitions"][0]
                    status = comp["status"]["type"]["name"]
                    if status not in ("STATUS_FINAL", "STATUS_FULL_TIME"):
                        continue
                    gid = event["id"]
                    all_game_ids.append(gid)
                    teams = comp.get("competitors", [])
                    home  = next((t["team"]["displayName"] for t in teams
                                  if t["homeAway"] == "home"), "")
                    away  = next((t["team"]["displayName"] for t in teams
                                  if t["homeAway"] == "away"), "")
                    game_meta_d1[gid] = {
                        "date":      str(current),
                        "home_team": home,
                        "away_team": away,
                    }

                day_count += len(events)
                time.sleep(0.1)

                # If ESPN returned fewer events than page size we have all of them
                if len(events) < PAGE_SIZE:
                    break
                offset += PAGE_SIZE

            except Exception as e:
                print(f"  Warning: scoreboard fetch failed for {date_str} "
                      f"offset={offset}: {e}")
                break

        if day_count > 0:
            print(f"  {date_str}: {day_count} games found")

        current += timedelta(days=1)

    # Deduplicate (some games appear on multiple date pages near midnight)
    seen = set()
    unique_ids = []
    for gid in all_game_ids:
        if gid not in seen:
            seen.add(gid)
            unique_ids.append(gid)
    all_game_ids = unique_ids

    # Filter out games we've already checkpointed
    todo = [gid for gid in all_game_ids if gid not in already_done]

    print(f"\nTotal D1 games found:    {len(all_game_ids):,}")
    print(f"Already checkpointed:    {len(already_done):,}")
    print(f"Remaining to scrape:     {len(todo):,}")
    print(f"\nEstimated time:          {len(todo)*0.9/60:.0f}–{len(todo)*1.5/60:.0f} minutes")
    print("Tip: run with   caffeinate -i python3 update_boxscores.py --all_d1")
    print("     to prevent your Mac from sleeping mid-scrape.\n")

    # ── Step 2: Scrape each game, save to checkpoint ──────────────────────────
    # We save each game as its own parquet file immediately after scraping.
    # If the script is interrupted at any point, all saved files are safe.
    # On the next run, those game IDs are in `already_done` and get skipped.

    failed_d1   = []
    saved_count = 0

    # Handle Ctrl+C gracefully — merge what we have instead of just dying
    interrupted = False
    def _handle_interrupt(sig, frame):
        global interrupted
        print("\n\nInterrupted! Merging checkpointed games before exiting...")
        interrupted = True
    signal.signal(signal.SIGINT, _handle_interrupt)

    for i, gid in enumerate(todo):
        if interrupted:
            break

        meta = game_meta_d1.get(gid, {})
        print(f"  [{i+1+len(already_done):>5}/{len(all_game_ids)}]  "
              f"{meta.get('date','')}  "
              f"{meta.get('away_team','')} @ {meta.get('home_team','')}")

        try:
            box = s.get_game_boxscore(gid)

            if box is None or box.empty:
                print("    -> empty, skipping")
                failed_d1.append(gid)
                continue

            box = box[box["player"] != "TEAM"].copy()
            box["game_date"]  = meta.get("date", "")
            box["home_team"]  = meta.get("home_team", "")
            box["away_team"]  = meta.get("away_team", "")

            # Save this single game to the checkpoint directory
            ckpt_path = os.path.join(CKPT_DIR, f"{gid}.parquet")
            box.to_parquet(ckpt_path, index=False)
            saved_count += 1

            time.sleep(0.8)

        except Exception as e:
            print(f"    -> Error: {e}")
            failed_d1.append(gid)
            time.sleep(2)

    # ── Step 3: Merge ALL season checkpoint folders into final output ────────
    # We scan every checkpoint_d1_* folder so that running this script for
    # multiple seasons (2024, 2025, 2026) accumulates into one combined file.
    # Each checkpoint file is tagged with the season it came from.

    print(f"\nMerging checkpoint files from all seasons...")

    season_folders = [
        d for d in os.listdir(args.output_dir)
        if d.startswith("checkpoint_d1_")
        and os.path.isdir(os.path.join(args.output_dir, d))
    ]

    all_frames = []
    per_season_counts = {}

    for folder in season_folders:
        # Extract season year from folder name "checkpoint_d1_2024" → 2024
        try:
            folder_season = int(folder.replace("checkpoint_d1_", ""))
        except ValueError:
            folder_season = None

        folder_path = os.path.join(args.output_dir, folder)
        ckpt_files = [f for f in os.listdir(folder_path) if f.endswith(".parquet")]
        season_frames = []

        for f in ckpt_files:
            try:
                df_game = pd.read_parquet(os.path.join(folder_path, f))
                if folder_season is not None and "season" not in df_game.columns:
                    df_game["season"] = folder_season
                season_frames.append(df_game)
            except Exception:
                pass

        if season_frames:
            season_df = pd.concat(season_frames, ignore_index=True)
            all_frames.append(season_df)
            per_season_counts[folder_season] = {
                "games": season_df["game_id"].nunique(),
                "rows":  len(season_df),
            }
            print(f"  {folder}: {len(ckpt_files):,} games loaded")

    if not all_frames:
        print("No checkpoint files found. Nothing to merge.")
        exit(1)

    combined_d1 = pd.concat(all_frames, ignore_index=True)

    # Deduplicate in case any game_id exists in multiple season folders
    combined_d1 = combined_d1.drop_duplicates(subset=["game_id", "player"],
                                                keep="last").reset_index(drop=True)

    out_parquet = os.path.join(args.output_dir, "ncaa_box_scores.parquet")
    combined_d1.to_parquet(out_parquet, index=False)

    print(f"\n  Per-season breakdown:")
    for yr in sorted(per_season_counts):
        stats = per_season_counts[yr]
        print(f"    {yr}: {stats['games']:,} games, {stats['rows']:,} rows")

    print(f"\n{'='*55}")
    print(f"  D1 scrape {'(partial) ' if interrupted else ''}complete")
    print(f"{'='*55}")
    print(f"  Games in output:    {combined_d1['game_id'].nunique():,}")
    print(f"  Player-game rows:   {len(combined_d1):,}")
    print(f"  Unique teams:       {combined_d1['team'].nunique():,}")
    print(f"  Failed this run:    {len(failed_d1)}")
    if interrupted:
        print(f"\n  Scrape was interrupted.")
        print(f"  Re-run the same command to resume from game {i+len(already_done)+1}.")
    if failed_d1:
        print(f"\n  Failed IDs saved to: {args.output_dir}/failed_games.json")
        with open(os.path.join(args.output_dir, "failed_games.json"), "w") as f:
            json.dump(failed_d1, f)
    print(f"\n  Output: {out_parquet}")
    exit()


# ── OPTION C: Multi-season NC A&T scrape ────────────────────────────────────
if args.multi_season:
    # Scrape 3 seasons of NC A&T data and stack them into one file.
    # This gives ~90 games and ~1,800 player-game rows — enough for the
    # model to actually beat the season-average baseline.
    #
    # Each season is scraped independently then tagged with a season column
    # so you can filter by year later if needed.
    #
    # Roster note: player names change across seasons (transfers, freshmen).
    # The model treats each player-season as independent which is correct —
    # a freshman's 2024 stats shouldn't directly predict a senior's 2026 stats.
    # Rolling averages are computed within-season only (handled in feature_engineering.py).

    SEASONS_TO_SCRAPE = [2024, 2025, 2026]
    all_season_frames = []

    for season_yr in SEASONS_TO_SCRAPE:
        print(f"\n{'='*55}")
        print(f"  Scraping NC A&T season {season_yr-1}-{str(season_yr)[-2:]}")
        print(f"{'='*55}")

        season_url = (
            f"https://site.api.espn.com/apis/site/v2/sports/basketball/"
            f"mens-college-basketball/teams/{NCAT_ESPN_ID}/schedule"
            f"?season={season_yr}"
        )
        try:
            r = requests.get(season_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            r.raise_for_status()
            sched = r.json()
        except Exception as e:
            print(f"  Could not fetch schedule for {season_yr}: {e}")
            continue

        season_game_ids  = []
        season_game_meta = {}

        for event in sched.get("events", []):
            comp   = event["competitions"][0]
            status = comp["status"]["type"]["name"]
            if status not in ("STATUS_FINAL", "STATUS_FULL_TIME"):
                continue

            gid = event["id"]
            season_game_ids.append(gid)

            ncat_score = opp_score = None
            opp_name   = "Unknown"
            home_away  = "home"

            for c in comp["competitors"]:
                is_ncat   = str(c["team"].get("id", "")) == NCAT_ESPN_ID
                raw_score = c.get("score", 0)
                if isinstance(raw_score, dict):
                    score = int(raw_score.get("value", raw_score.get("displayValue", 0)))
                else:
                    score = int(raw_score) if raw_score else 0
                if is_ncat:
                    ncat_score = score
                    home_away  = c["homeAway"]
                else:
                    opp_score = score
                    opp_name  = c["team"].get("displayName", "Unknown")

            season_game_meta[gid] = {
                "date":       event.get("date", "")[:10],
                "opponent":   opp_name,
                "home_away":  home_away,
                "ncat_score": ncat_score,
                "opp_score":  opp_score,
                "win":        1 if (ncat_score or 0) > (opp_score or 0) else 0,
                "season":     season_yr,
            }

        print(f"  Found {len(season_game_ids)} completed games.")

        # Scrape each game
        season_boxes = []
        for i, gid in enumerate(season_game_ids):
            meta = season_game_meta[gid]
            print(f"  [{i+1:>2}/{len(season_game_ids)}] {meta['date']}  "
                  f"NC A&T vs {meta['opponent']}  "
                  f"({meta['ncat_score']}-{meta['opp_score']})")
            try:
                box = s.get_game_boxscore(gid)
                if box is None or box.empty:
                    print("    -> empty, skipping")
                    continue
                box = box[box["player"] != "TEAM"].copy()
                box["game_date"]  = meta["date"]
                box["opponent"]   = meta["opponent"]
                box["home_away"]  = meta["home_away"]
                box["ncat_score"] = meta["ncat_score"]
                box["opp_score"]  = meta["opp_score"]
                box["win"]        = meta["win"]
                box["season"]     = meta["season"]   # ← tag each row with the season year
                season_boxes.append(box)
                time.sleep(0.8)
            except Exception as e:
                print(f"    -> Error: {e}")
                time.sleep(2)

        if season_boxes:
            season_df = pd.concat(season_boxes, ignore_index=True)
            all_season_frames.append(season_df)
            print(f"  Season {season_yr}: {len(season_boxes)} games, "
                  f"{len(season_df):,} player rows scraped.")
        else:
            print(f"  Season {season_yr}: no data collected.")

    if not all_season_frames:
        print("\nNo data collected across any season. Check your connection.")
        exit(1)

    # Combine all seasons
    combined_multi = pd.concat(all_season_frames, ignore_index=True)

    # Reorder columns
    front_cols = ["season", "game_id", "game_date", "opponent", "home_away",
                  "ncat_score", "opp_score", "win",
                  "team", "player", "player_id", "position", "starter",
                  "min", "pts", "reb", "ast", "stl", "blk", "to",
                  "fgm", "fga", "2pm", "2pa", "3pm", "3pa",
                  "ftm", "fta", "oreb", "dreb", "pf"]
    extra_cols = [c for c in combined_multi.columns if c not in front_cols]
    combined_multi = combined_multi[[c for c in front_cols if c in combined_multi.columns] + extra_cols]

    # Save — overwrites games_raw so feature_engineering.py picks it up automatically
    parquet_path = os.path.join(args.output_dir, "games_raw.parquet")
    csv_path     = os.path.join(args.output_dir, "games_raw.csv")
    combined_multi.to_parquet(parquet_path, index=False)
    combined_multi.to_csv(csv_path, index=False)

    ncat_multi = combined_multi[combined_multi["team"].str.contains("A&T", na=False)]

    print(f"\n{'='*55}")
    print(f"  Multi-season scrape complete")
    print(f"{'='*55}")
    for yr in SEASONS_TO_SCRAPE:
        yr_rows = ncat_multi[ncat_multi["season"] == yr]
        games   = yr_rows["game_id"].nunique()
        print(f"  Season {yr}: {games} games, {len(yr_rows):,} NC A&T player rows")
    print(f"\n  Combined total:")
    print(f"    Total player rows:  {len(combined_multi):,}")
    print(f"    NC A&T rows:        {len(ncat_multi):,}")
    print(f"    Unique players:     {ncat_multi['player'].nunique()}")
    print(f"\n  Saved to:")
    print(f"    {parquet_path}")
    print(f"    {csv_path}")
    print(f"\n  Next steps:")
    print(f"    python3 feature_engineering.py")
    print(f"    python3 model_player_performance.py")
    exit()


# ── OPTION B: NC A&T only (default, fast) ────────────────────────────────────

# --- 1. GET GAME IDs FROM ESPN SCHEDULE API -----------------------------------
# ESPN exposes a free, undocumented but stable JSON API for team schedules.
# No key needed. Returns every game (past and future) for the season.

print(f"Fetching NC A&T schedule from ESPN for season {args.season}...")

schedule_url = (
    f"https://site.api.espn.com/apis/site/v2/sports/basketball/"
    f"mens-college-basketball/teams/{NCAT_ESPN_ID}/schedule"
    f"?season={args.season}"
)

resp = requests.get(schedule_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
resp.raise_for_status()
schedule_json = resp.json()

# Walk the JSON to collect completed game IDs and basic metadata
game_ids   = []
game_meta  = {}   # game_id -> {opponent, date, home_away, ncat_score, opp_score}

for event in schedule_json.get("events", []):
    competition = event["competitions"][0]

    # Skip games that haven't been played yet
    status = competition["status"]["type"]["name"]
    if status not in ("STATUS_FINAL", "STATUS_FULL_TIME"):
        continue

    game_id = event["id"]
    game_ids.append(game_id)

    # Figure out opponent name and scores
    ncat_score = opp_score = None
    opp_name   = "Unknown"
    home_away  = "home"

    for comp in competition["competitors"]:
        is_ncat = str(comp["team"].get("id", "")) == NCAT_ESPN_ID
        raw_score = comp.get("score", 0)
        # ESPN returns score as either a plain number OR {"value": 82, "displayValue": "82"}
        if isinstance(raw_score, dict):
            score = int(raw_score.get("value", raw_score.get("displayValue", 0)))
        else:
            score = int(raw_score) if raw_score else 0
        if is_ncat:
            ncat_score = score
            home_away  = comp["homeAway"]
        else:
            opp_score = score
            opp_name  = comp["team"].get("displayName", "Unknown")

    game_meta[game_id] = {
        "date":       event.get("date", "")[:10],
        "opponent":   opp_name,
        "home_away":  home_away,
        "ncat_score": ncat_score,
        "opp_score":  opp_score,
        "win":        1 if (ncat_score or 0) > (opp_score or 0) else 0,
    }

print(f"Found {len(game_ids)} completed games.\n")


# --- 2. SCRAPE EACH BOX SCORE -------------------------------------------------
# Known working columns (confirmed from test run):
# game_id, team, player, player_id, position, starter, min,
# fgm, fga, 2pm, 2pa, 3pm, 3pa, ftm, fta,
# pts, reb, ast, to, stl, blk, oreb, dreb, pf

all_boxes   = []
failed_ids  = []

for i, game_id in enumerate(game_ids):
    meta = game_meta[game_id]
    print(f"  [{i+1}/{len(game_ids)}] {meta['date']}  NC A&T vs {meta['opponent']}"
          f"  ({meta['ncat_score']}-{meta['opp_score']})")
    try:
        box = s.get_game_boxscore(game_id)

        # Skip if CBBpy returned empty (game data not available)
        if box is None or box.empty:
            print("    -> Empty box score, skipping.")
            failed_ids.append(game_id)
            continue

        # Drop the synthetic "TEAM" totals row (player == "TEAM", no player_id)
        box = box[box["player"] != "TEAM"].copy()

        # Attach the metadata we already have from the schedule API
        box["game_date"]  = meta["date"]
        box["opponent"]   = meta["opponent"]
        box["home_away"]  = meta["home_away"]
        box["ncat_score"] = meta["ncat_score"]
        box["opp_score"]  = meta["opp_score"]
        box["win"]        = meta["win"]

        all_boxes.append(box)

        # Be polite to ESPN — short pause between requests
        time.sleep(0.8)

    except Exception as e:
        print(f"    -> Error: {e}")
        failed_ids.append(game_id)
        time.sleep(2)   # longer pause after an error


# --- 3. COMBINE AND SAVE ------------------------------------------------------
if not all_boxes:
    print("\nNo box scores collected. Check your internet connection or game IDs.")
    exit(1)

combined = pd.concat(all_boxes, ignore_index=True)

# Reorder columns so the most useful ones come first
front_cols = ["game_id", "game_date", "opponent", "home_away",
              "ncat_score", "opp_score", "win",
              "team", "player", "player_id", "position", "starter",
              "min", "pts", "reb", "ast", "stl", "blk", "to",
              "fgm", "fga", "2pm", "2pa", "3pm", "3pa",
              "ftm", "fta", "oreb", "dreb", "pf"]
extra_cols = [c for c in combined.columns if c not in front_cols]
combined   = combined[front_cols + extra_cols]

# Save as both parquet (fast/small) and CSV (dashboard-friendly)
parquet_path = os.path.join(args.output_dir, "games_raw.parquet")
csv_path     = os.path.join(args.output_dir, "games_raw.csv")
combined.to_parquet(parquet_path, index=False)
combined.to_csv(csv_path, index=False)

# Also save a legacy games.json so your existing dashboard still works
ncat_only = combined[combined["team"].str.contains("A&T", na=False)]
games_json = {}
for gid, grp in ncat_only.groupby("game_id"):
    meta = game_meta.get(gid, {})
    games_json[f"game_{gid}"] = {
        "name":        meta.get("opponent", "Unknown"),
        "date":        meta.get("date", ""),
        "ncat_score":  meta.get("ncat_score"),
        "opp_score":   meta.get("opp_score"),
        "win":         meta.get("win"),
        "ncat_players": grp[["player", "min", "pts", "reb", "ast",
                               "stl", "blk", "to", "pf"]].to_dict("records"),
        "ncat_totals": {
            "PTS": meta.get("ncat_score", 0),
            "REB": int(grp["reb"].sum()),
            "AST": int(grp["ast"].sum()),
            "TO":  int(grp["to"].sum()),
            "STL": int(grp["stl"].sum()),
            "BLK": int(grp["blk"].sum()),
        },
        "stats": ["PTS", "REB", "AST", "STL", "BLK", "TO"]
    }

json_path = os.path.join(args.output_dir, "games.json")
with open(json_path, "w") as f:
    json.dump(games_json, f, indent=2)


# --- 4. SUMMARY ---------------------------------------------------------------
print(f"\n{'='*55}")
print(f"  Scrape complete")
print(f"{'='*55}")
print(f"  Games scraped:      {len(all_boxes)}")
print(f"  Failed games:       {len(failed_ids)}")
print(f"  Total player rows:  {len(combined):,}")
print(f"  NC A&T players:     {ncat_only['player'].nunique()}")
print(f"\n  Saved to:")
print(f"    {parquet_path}")
print(f"    {csv_path}")
print(f"    {json_path}")

# Quick preview — NC A&T top scorers this season
print(f"\n  NC A&T top scorers this season (avg pts/game):")
top = (ncat_only[ncat_only["min"] > 0]
       .groupby("player")["pts"]
       .mean()
       .sort_values(ascending=False)
       .head(8)
       .round(1))
for player, avg in top.items():
    print(f"    {player:<25} {avg} ppg")

if failed_ids:
    print(f"\n  Failed game IDs (can retry individually): {failed_ids}")
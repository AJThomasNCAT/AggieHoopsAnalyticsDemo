"""
model_player_performance.py
---------------------------
Predicts NC A&T player performance using pre-game features.
Now includes opponent difficulty (opp_drtg_pregame) from D1 data.

Three models:
  1. Regression  → pts_per40
  2. Regression  → ts_pct
  3. Classifier  → will player exceed their season avg pts_per40?

Run:
    python3 model_player_performance.py
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

OUT_DIR     = "assets/data"
PLAYER_PATH = os.path.join(OUT_DIR, "player_advanced.csv")

if not os.path.exists(PLAYER_PATH):
    raise SystemExit(f"{PLAYER_PATH} missing. Run feature_engineering.py first.")

player = pd.read_csv(PLAYER_PATH, parse_dates=["game_date"])

# NC A&T only, meaningful minutes
ncat = player[
    player["team"].str.contains("A&T", na=False) &
    (player["min"] >= 10)
].copy()

print(f"NC A&T player-game rows (10+ min): {len(ncat):,}")
print(f"Unique players: {ncat['player'].nunique()}")
print(f"Players: {sorted(ncat['player'].unique())}\n")

opp_coverage = ncat["opp_drtg_pregame"].notna().mean()
print(f"Opponent difficulty coverage: {opp_coverage*100:.0f}% of rows have opp_drtg_pregame\n")


# ── Feature sets ──────────────────────────────────────────────────────────────
# We build two feature sets:
#   FEATURES_BASE  — works for all rows (no opponent rating needed)
#   FEATURES_FULL  — adds opponent difficulty (only rows with D1 match)
#
# We train and report both so you can see how much the opponent feature helps.

FEATURES_BASE = [
    # Recent form (rolling window)
    "ts_pct_r5",          # rolling true shooting % — recent efficiency
    "usg_pct_r5",         # rolling usage rate — recent involvement
    "pts_per40_r5",       # rolling pts per 40 — recent scoring
    "reb_per40_r5",       # rolling rebounds per 40 — NEW
    "ast_per40_r5",       # rolling assists per 40
    # Season baselines
    "ts_pct_season",      # season avg TS% — baseline efficiency
    "pts_per40_season",   # season avg pts/40 — baseline scoring
    "reb_per40_season",   # season avg reb/40 — NEW
    "to_per40_season",    # season avg to/40 — NEW (turnovers beat baseline)
    # Context
    "days_rest",          # fatigue
    "is_home",            # home court
    "on_hot_streak",      # NEW: trending up in last 2 games?
    "position_enc",       # NEW: 0=Guard, 1=Wing, 2=Big
]

FEATURES_FULL = FEATURES_BASE + [
    "opp_drtg_pregame",   # Torvik adjde (iterated, SOS-adjusted defense)
    "opp_pace_pregame",   # Torvik adjt (opponent tempo)
    "torvik_barthag",     # Torvik composite opponent strength
    "torvik_adjoe",       # opponent offensive efficiency
    # PBP shot quality features (from scrape_pbp_features.py)
    # Rolling 5-game averages — pregame only (no current-game leakage)
    "rim_rate_r5",        # how often player attacks the basket recently
    "three_rate_r5",      # shooting profile trend
    "avg_shot_dist_r5",   # shot quality proxy (lower = better)
    "assisted_rate_r5",   # playmaking dependency
    "foul_drawn_rate_r5", # floor-raiser ability
    # Opponent shot profile (how does this defense actually defend)
    "opp_rim_rate_r5",    # does this opponent give up rim attempts?
    "opp_three_rate_r5",  # does this defense force 3s?
]

TARGET_PTS = "pts_per40"
TARGET_TS  = "ts_pct"


def run_models(df, features, label):
    """Train and evaluate all three models on a given dataframe + feature set."""

    df = df.dropna(subset=features + [TARGET_PTS, TARGET_TS]).copy()

    # Drop players with fewer than 4 rows after filtering
    counts = df.groupby("player").size()
    df = df[df["player"].isin(counts[counts >= 4].index)].reset_index(drop=True)

    if len(df) < 20:
        print(f"  [{label}] Not enough rows ({len(df)}) to train. Skipping.\n")
        return None

    df["over_avg"] = (df[TARGET_PTS] > df["pts_per40_season"]).astype(int)

    # Time-ordered split
    df = df.sort_values("game_date").reset_index(drop=True)
    cutoff   = int(len(df) * 0.80)
    train    = df.iloc[:cutoff]
    test     = df.iloc[cutoff:]

    X_train, X_test = train[features], test[features]

    print(f"\n{'─'*52}")
    print(f"  Feature set: {label}")
    print(f"  Rows: {len(df)}  |  Train: {len(train)}  |  Test: {len(test)}")
    print(f"  Train up to: {train['game_date'].max().date()}")
    print(f"{'─'*52}")

    # Model 1: pts_per40 regression
    reg_pts = GradientBoostingRegressor(
        n_estimators=200, max_depth=2, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    reg_pts.fit(X_train, train[TARGET_PTS])
    pred_pts     = reg_pts.predict(X_test)
    mae_model    = mean_absolute_error(test[TARGET_PTS], pred_pts)
    mae_baseline = mean_absolute_error(test[TARGET_PTS], test["pts_per40_season"])
    r2           = r2_score(test[TARGET_PTS], pred_pts)

    print(f"\n  pts_per40 model:")
    print(f"    MAE:           {mae_model:.2f} pts/40")
    print(f"    Baseline MAE:  {mae_baseline:.2f} pts/40  (predict season avg)")
    print(f"    R²:            {r2:.3f}")
    diff = mae_baseline - mae_model
    print(f"    Model is {'BETTER' if diff > 0 else 'WORSE'} than baseline "
          f"by {abs(diff):.2f} pts/40")

    # Model 2: ts_pct regression
    reg_ts = GradientBoostingRegressor(
        n_estimators=200, max_depth=2, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    reg_ts.fit(X_train, train[TARGET_TS])
    pred_ts = reg_ts.predict(X_test)
    mae_ts  = mean_absolute_error(test[TARGET_TS], pred_ts)
    r2_ts   = r2_score(test[TARGET_TS], pred_ts)
    print(f"\n  ts_pct model:")
    print(f"    MAE:  {mae_ts:.3f}  (±{mae_ts*100:.1f} pct points)")
    print(f"    R²:   {r2_ts:.3f}")

    # Model 3: above-avg classifier
    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=2, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    clf.fit(X_train, train["over_avg"])
    pred_cls  = clf.predict(X_test)
    acc       = accuracy_score(test["over_avg"], pred_cls)
    base_rate = max(test["over_avg"].mean(), 1 - test["over_avg"].mean())
    print(f"\n  above-avg classifier:")
    print(f"    Accuracy:  {acc*100:.1f}%")
    print(f"    Baseline:  {base_rate*100:.1f}%  (majority class)")

    # Feature importance
    print(f"\n  Feature importance (pts_per40 model):")
    imp = sorted(zip(features, reg_pts.feature_importances_),
                 key=lambda x: x[1], reverse=True)
    for feat, val in imp:
        bar = "█" * int(round(val * 40))
        print(f"    {feat:<25} {val*100:5.1f}%  {bar}")

    return {
        "label":          label,
        "n_rows":         len(df),
        "mae_pts_model":  round(float(mae_model),    2),
        "mae_pts_base":   round(float(mae_baseline), 2),
        "r2_pts":         round(float(r2),           3),
        "mae_ts":         round(float(mae_ts),       4),
        "acc_over_avg":   round(float(acc * 100),    1),
        "feature_importance": [
            {"feature": f, "importance_pct": round(float(v * 100), 2)}
            for f, v in imp
        ],
        "train_cutoff":   str(train["game_date"].max().date()),
        "pred_pts":       pred_pts,
        "pred_ts":        pred_ts,
        "pred_cls":       pred_cls,
        "test_df":        test,
        "reg_pts":        reg_pts,
    }


# ── Run base model (all rows) ─────────────────────────────────────────────────
base_result = run_models(ncat.copy(), FEATURES_BASE, "BASE (no opponent rating)")

# ── Run full model (rows with opponent rating) ────────────────────────────────
# Drop any FEATURES_FULL columns that don't exist in the data yet
# (e.g. PBP features before scrape_pbp_features.py has been run)
available_full = [f for f in FEATURES_FULL if f in ncat.columns]
missing_full   = [f for f in FEATURES_FULL if f not in ncat.columns]
if missing_full:
    print(f"\n  ⚠  Dropping {len(missing_full)} features not yet in data:")
    for f in missing_full:
        print(f"       – {f}")
    print(f"     Run scrape_pbp_features.py then feature_engineering.py to add them.\n")

ncat_with_opp = ncat[ncat["opp_drtg_pregame"].notna()].copy()
full_result   = run_models(ncat_with_opp, available_full,
                           "FULL (with opponent difficulty)")


# ── Per-player season summary ─────────────────────────────────────────────────
print(f"\n{'─'*52}")
print(f"  Per-player season summary")
print(f"{'─'*52}")

summary = (ncat.groupby("player")
               .agg(
                   games         = ("game_date", "count"),
                   avg_min       = ("min",         "mean"),
                   avg_pts       = ("pts",         "mean"),
                   avg_reb       = ("reb",         "mean"),
                   avg_ast       = ("ast",         "mean"),
                   avg_pts_per40 = ("pts_per40",   "mean"),
                   avg_ts_pct    = ("ts_pct",      "mean"),
                   avg_usg_pct   = ("usg_pct",     "mean"),
                   avg_reb_per40 = ("reb_per40",   "mean"),
                   avg_ast_per40 = ("ast_per40",   "mean"),
               )
               .round(2)
               .sort_values("avg_pts_per40", ascending=False))

print(summary.to_string())

print(f"\n{'─'*52}")
print(f"  Efficiency vs Volume  (advisor's point in numbers)")
print(f"{'─'*52}")
ev = summary[["avg_min", "avg_pts", "avg_pts_per40", "avg_ts_pct"]].copy()
ev["per40_rank"] = ev["avg_pts_per40"].rank(ascending=False).astype(int)
ev["raw_rank"]   = ev["avg_pts"].rank(ascending=False).astype(int)
ev["rank_diff"]  = ev["raw_rank"] - ev["per40_rank"]
print(ev.sort_values("per40_rank").to_string())
print("\n  + rank_diff = ranks HIGHER per-40 than raw pts (underutilised)")
print("  - rank_diff = ranks LOWER per-40 than raw pts (over-relied on)\n")


# ── Per-player prediction accuracy (test set only) ───────────────────────────
# For each rotation player with at least 3 test-set games, show:
#   - actual avg pts_per40 on the held-out games
#   - predicted avg pts_per40 from the model
#   - MAE (how far off the model was, on average)
#   - bias (positive = model over-predicted, negative = model under-predicted)
#
# This is the chart-ready output: players where MAE is low = model works well
# for them (usually consistent rotation players). Players with high MAE =
# model struggles (usually bench players with volatile roles).

print(f"{'─'*52}")
print(f"  Per-player prediction accuracy  (test set only)")
print(f"{'─'*52}")

# Pull test set + predictions from the best-performing model
best = full_result if full_result else base_result

if best is None:
    print("  No model was trained. Skipping per-player analysis.")
else:
    test_df = best["test_df"].copy()
    test_df["predicted_pts_per40"] = best["pred_pts"]

    # Compute per-player stats on the test set
    test_df["abs_error"] = (test_df["pts_per40"] - test_df["predicted_pts_per40"]).abs()
    test_df["bias"]      = test_df["predicted_pts_per40"] - test_df["pts_per40"]

    per_player = (test_df.groupby("player")
                         .agg(
                             test_games    = ("game_date",          "count"),
                             actual_avg    = ("pts_per40",           "mean"),
                             predicted_avg = ("predicted_pts_per40", "mean"),
                             mae           = ("abs_error",           "mean"),
                             bias          = ("bias",                "mean"),
                         )
                         .round(2))

    # Only show players with at least 3 test games (statistics aren't meaningful
    # on fewer games, and single-game variance dominates)
    per_player = per_player[per_player["test_games"] >= 3].copy()

    # Rank by actual avg pts_per40 so top scorers appear first
    per_player = per_player.sort_values("actual_avg", ascending=False)

    # Print the table
    print(f"  Showing players with 3+ games in test set:")
    print(f"  (using {best['label']} model — train size {len(best['test_df'])*4} rows)\n")

    # Format the header manually for better alignment
    print(f"  {'player':<22} {'test_games':>10} {'actual':>8} "
          f"{'predicted':>10} {'MAE':>6} {'bias':>7}")
    print(f"  {'─'*22} {'─'*10} {'─'*8} {'─'*10} {'─'*6} {'─'*7}")
    for p, row in per_player.iterrows():
        flag = ""
        if row["mae"] < 3.0:       flag = "  ← model nails it"
        elif row["mae"] > 7.0:     flag = "  ← model struggles"
        bias_sign = "+" if row["bias"] >= 0 else ""
        print(f"  {p:<22} {int(row['test_games']):>10} "
              f"{row['actual_avg']:>8.2f} {row['predicted_avg']:>10.2f} "
              f"{row['mae']:>6.2f} {bias_sign}{row['bias']:>6.2f}{flag}")

    # Summary interpretation
    print(f"\n  Interpretation:")
    print(f"    actual    = player's true avg pts/40 across their test games")
    print(f"    predicted = model's avg prediction for those games")
    print(f"    MAE       = how far off the model was on each game, averaged")
    print(f"    bias      = systematic over (+) or under (-) prediction")

    # Find the clearest "model works here" and "model fails here" cases
    if len(per_player) >= 2:
        best_player  = per_player["mae"].idxmin()
        worst_player = per_player["mae"].idxmax()
        print(f"\n  Most accurate prediction:  {best_player} "
              f"(MAE {per_player.loc[best_player, 'mae']:.2f})")
        print(f"  Least accurate prediction: {worst_player} "
              f"(MAE {per_player.loc[worst_player, 'mae']:.2f})")

    # Save for the dashboard — this is the JSON the chart will read from
    chart_data = per_player.reset_index().to_dict("records")
    with open(os.path.join(OUT_DIR, "player_prediction_accuracy.json"), "w") as f:
        json.dump({
            "model_used":   best["label"],
            "train_rows":   len(best["test_df"]) * 4,
            "test_rows":    len(test_df),
            "overall_mae":  round(float(test_df["abs_error"].mean()), 2),
            "per_player":   chart_data,
        }, f, indent=2)

    print(f"\n✅  player_prediction_accuracy.json → {OUT_DIR}/")
    print(f"    (Ready for dashboard visualization)")


# ── Export ────────────────────────────────────────────────────────────────────
# (best is already defined in the per-player analysis section above)

if best:
    test_df  = best["test_df"]
    preds_df = test_df[["game_date", "player", "team", "opponent",
                         TARGET_PTS, TARGET_TS, "pts_per40_season"]].copy()
    preds_df["predicted_pts_per40"] = best["pred_pts"].round(2)
    preds_df["predicted_ts_pct"]    = best["pred_ts"].round(3)
    preds_df["predicted_over_avg"]  = best["pred_cls"]
    preds_df.to_csv(os.path.join(OUT_DIR, "player_predictions.csv"), index=False)

    export = {
        "base_model":  {k: v for k, v in base_result.items()
                        if k not in ("pred_pts","pred_ts","pred_cls","test_df","reg_pts")}
                        if base_result else None,
        "full_model":  {k: v for k, v in full_result.items()
                        if k not in ("pred_pts","pred_ts","pred_cls","test_df","reg_pts")}
                        if full_result else None,
        "player_summary": summary.reset_index().to_dict("records"),
    }
    with open(os.path.join(OUT_DIR, "player_model_results.json"), "w") as f:
        json.dump(export, f, indent=2)

    print(f"✅  player_predictions.csv      → {OUT_DIR}/")
    print(f"✅  player_model_results.json   → {OUT_DIR}/")


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-STAT PROJECTION MODELS
# ═══════════════════════════════════════════════════════════════════════════════
# Train one gradient boosting regressor per stat (rebounds, assists, turnovers,
# steals, blocks) using the same pre-game features as the pts_per40 model.
# Export a per-player projection card showing all predicted stats at once.
#
# Honest note on stat predictability (documented for the writeup):
#   - reb_per40: most stable non-scoring stat, model should beat baseline
#   - ast_per40: stable for point guards, noisier for wings
#   - to_per40:  tied to usage rate, moderately predictable
#   - stl_per40: noisy at game level but meaningful season-long
#   - blk_per40: rare events, very noisy — model will struggle, expected
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*60}")
print(f"  MULTI-STAT PROJECTION MODELS")
print(f"{'═'*60}")

STAT_TARGETS = {
    "pts_per40": "Points per 40",
    "reb_per40": "Rebounds per 40",
    "ast_per40": "Assists per 40",
    "to_per40":  "Turnovers per 40",
    "stl_per40": "Steals per 40",
    "blk_per40": "Blocks per 40",
}

# Use the FULL feature set if available; otherwise BASE
multi_features = FEATURES_FULL
multi_source   = ncat.copy()

# ── Impute NaN for opponent features BEFORE dropna ───────────────────────────
# torvik_barthag and opp_drtg_pregame may be NaN for some rows (non-D1
# opponents, 2024/2025 games without Torvik data). Rather than dropping
# those rows entirely, impute with median. This keeps the full 3-season
# dataset available for training.
for col in ["opp_drtg_pregame", "opp_pace_pregame", "torvik_barthag"]:
    if col in multi_source.columns:
        median_val = multi_source[col].median()
        if pd.isna(median_val):
            # column is ALL NaN — remove it from feature set and fall back
            multi_features = [f for f in multi_features if f != col]
            print(f"  ⚠  {col} is all NaN — removed from feature set")
        else:
            multi_source[col] = multi_source[col].fillna(median_val)
    else:
        # column doesn't exist in data yet — remove from feature set
        multi_features = [f for f in multi_features if f != col]
        print(f"  ⚠  {col} not in data — run feature_engineering.py first. Removed.")

# If we've lost all opponent features, fall back entirely to BASE
opp_features_present = any(
    f in multi_features for f in ["opp_drtg_pregame", "opp_pace_pregame", "torvik_barthag"]
)
if not opp_features_present:
    print("  Falling back to BASE features (no opponent data available).")
    multi_features = FEATURES_BASE

# Drop rows missing any feature
multi_df = multi_source.dropna(subset=multi_features + list(STAT_TARGETS.keys())).copy()

# Only keep players with 4+ games after filtering
counts = multi_df.groupby("player").size()
multi_df = multi_df[multi_df["player"].isin(counts[counts >= 4].index)]
multi_df = multi_df.sort_values("game_date").reset_index(drop=True)

# Time-ordered split
cutoff = int(len(multi_df) * 0.80)
train  = multi_df.iloc[:cutoff]
test   = multi_df.iloc[cutoff:]

print(f"  Using feature set: {'FULL' if 'opp_drtg_pregame' in multi_features else 'BASE'}")
print(f"  Rows: {len(multi_df)}  |  Train: {len(train)}  |  Test: {len(test)}\n")

X_train, X_test = train[multi_features], test[multi_features]

# Train one regressor per stat
stat_models   = {}  # stat_name → trained model
stat_metrics  = {}  # stat_name → {mae, baseline_mae, r2}
stat_predictions = {}  # stat_name → test predictions

print(f"  {'Stat':<22} {'Model MAE':>10} {'Baseline':>10} {'R²':>7} {'Beats?':>8}")
print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*7} {'─'*8}")

for stat_col, stat_name in STAT_TARGETS.items():
    # Train
    mdl = GradientBoostingRegressor(
        n_estimators=200, max_depth=2, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    mdl.fit(X_train, train[stat_col])
    preds = mdl.predict(X_test)

    # Baseline = predict the player's season average (requires a season-avg column)
    # We build it on the fly for any stat
    season_col = f"{stat_col}_season_avg"
    player_season_avg = (train.groupby("player")[stat_col].mean().to_dict())
    test_baseline = test["player"].map(player_season_avg).fillna(train[stat_col].mean())

    mae_model = mean_absolute_error(test[stat_col], preds)
    mae_base  = mean_absolute_error(test[stat_col], test_baseline)
    r2        = r2_score(test[stat_col], preds)

    stat_models[stat_col]      = mdl
    stat_metrics[stat_col]     = {
        "stat_display": stat_name,
        "mae_model":    round(float(mae_model), 3),
        "mae_baseline": round(float(mae_base),  3),
        "r2":           round(float(r2),        3),
        "beats_baseline": bool(mae_model < mae_base),
    }
    stat_predictions[stat_col] = preds

    flag = "✓ YES" if mae_model < mae_base else "✗ no"
    print(f"  {stat_name:<22} {mae_model:>10.2f} {mae_base:>10.2f} {r2:>7.3f} {flag:>8}")


# ── Per-player projections (next game) ───────────────────────────────────────
# For each player, generate a projection card showing their predicted stats
# for their most recent pregame feature row (this is what a coach would see
# right before the next tip-off).

print(f"\n  Generating per-player projection cards...")

# Get each player's most recent row of pre-game features
latest_per_player = (multi_df.sort_values("game_date")
                             .groupby("player")
                             .tail(1)
                             .reset_index(drop=True))

projection_cards = []

for _, row in latest_per_player.iterrows():
    # Skip if this player doesn't have enough training history
    if row["player"] not in train["player"].values:
        continue

    player_features = pd.DataFrame(
        row[multi_features].values.reshape(1, -1),
        columns=multi_features
    )

    # Predict each stat for this player's next game
    projections = {}
    for stat_col, mdl in stat_models.items():
        pred = float(mdl.predict(player_features)[0])
        projections[stat_col] = round(pred, 2)

    # Also compute their actual season averages for comparison
    player_all_games = multi_df[multi_df["player"] == row["player"]]
    season_avgs = {
        stat: round(float(player_all_games[stat].mean()), 2)
        for stat in STAT_TARGETS
    }

    # Usage and efficiency for context
    ts_pct  = round(float(player_all_games["ts_pct"].mean()),  3)
    usg_pct = round(float(player_all_games["usg_pct"].mean()), 2)
    avg_min = round(float(player_all_games["min"].mean()),     1)
    games   = int(len(player_all_games))

    projection_cards.append({
        "player":        row["player"],
        "games_played":  games,
        "avg_minutes":   avg_min,
        "season_avg_ts_pct":  ts_pct,
        "season_avg_usg_pct": usg_pct,
        "projections": {
            stat: {
                "predicted":  projections[stat],
                "season_avg": season_avgs[stat],
                "display":    STAT_TARGETS[stat],
            }
            for stat in STAT_TARGETS
        },
        "last_game_date": row["game_date"].strftime("%Y-%m-%d"),
        "last_opponent":  str(row.get("opponent", "Unknown")),
    })

# Sort projections by predicted pts_per40 descending
projection_cards.sort(
    key=lambda c: c["projections"]["pts_per40"]["predicted"],
    reverse=True
)

# Write the multi-stat export
multi_export = {
    "feature_set":     "FULL" if "opp_drtg_pregame" in multi_features else "BASE",
    "train_rows":      int(len(train)),
    "test_rows":       int(len(test)),
    "train_cutoff":    str(train["game_date"].max().date()),
    "stat_metrics":    stat_metrics,
    "projection_cards": projection_cards,
}

multi_path = os.path.join(OUT_DIR, "player_multi_stat_projections.json")
with open(multi_path, "w") as f:
    json.dump(multi_export, f, indent=2)

print(f"\n✅  player_multi_stat_projections.json → {OUT_DIR}/")
print(f"    {len(projection_cards)} player projection cards exported")
print(f"    Each card contains predictions + season averages for all 6 stats")
"""
model_player_performance_womens.py
-----------------------------------
Identical to model_player_performance.py but for NC A&T Women's Basketball.
Reads player_advanced_womens.csv and exports:
  player_model_results_womens.json
  player_prediction_accuracy_womens.json
  player_multi_stat_projections_womens.json

Run:
    python3 model_player_performance_womens.py
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics  import mean_absolute_error, r2_score, accuracy_score

OUT_DIR    = "assets/data"
INPUT_PATH = os.path.join(OUT_DIR, "player_advanced_womens.csv")

if not os.path.exists(INPUT_PATH):
    raise SystemExit(
        f"\n{INPUT_PATH} not found.\n"
        f"Run python3 feature_engineering_womens.py first.\n"
    )

# ── Feature sets (mirrors men's exactly) ─────────────────────────────────────
FEATURES_BASE = [
    "ts_pct_r5", "usg_pct_r5", "pts_per40_r5", "reb_per40_r5", "ast_per40_r5",
    "ts_pct_season", "pts_per40_season", "reb_per40_season", "to_per40_season",
    "days_rest", "is_home", "on_hot_streak", "position_enc",
]

# Women's FULL feature set is leaner than men's — 720 rows with 22 features
# causes overfitting. We keep the 4 most impactful opponent features only.
# PBP features added only if the merge succeeded (checked at runtime).
FEATURES_FULL = FEATURES_BASE + [
    "opp_drtg_pregame",   # opponent defensive efficiency
    "opp_pace_pregame",   # opponent tempo
    "torvik_barthag",     # composite opponent strength
    "torvik_adjoe",       # opponent offensive efficiency
]

# PBP features added dynamically below if merge succeeded
PBP_FEATURES = [
    "rim_rate_r5", "three_rate_r5", "avg_shot_dist_r5",
    "assisted_rate_r5", "foul_drawn_rate_r5",
]

TARGET_PTS = "pts_per40"
TARGET_TS  = "ts_pct"

# ── Load data ─────────────────────────────────────────────────────────────────
df_raw = pd.read_csv(INPUT_PATH, parse_dates=["game_date"])
df_raw = df_raw.sort_values("game_date").reset_index(drop=True)

# Filter to NC A&T women's players with 10+ minutes
ncat = df_raw[
    df_raw["team"].str.contains("A&T", na=False) &
    (df_raw["min"] >= 10)
].copy()

print(f"NC A&T women's player-game rows (10+ min): {len(ncat)}")
print(f"Unique players: {ncat['player'].nunique()}")
print(f"Players: {sorted(ncat['player'].unique().tolist())}\n")

opp_coverage = ncat["opp_drtg_pregame"].notna().mean() * 100
print(f"Opponent difficulty coverage: {opp_coverage:.0f}% of rows\n")


# ── Model runner (identical to men's) ────────────────────────────────────────
def run_models(df, features, label):
    available = [f for f in features if f in df.columns]
    missing   = [f for f in features if f not in df.columns]
    if missing:
        print(f"  ⚠  Dropping {len(missing)} unavailable features: {missing[:4]}...")

    # Also remove features that are all NaN — they'd drop every row in dropna
    all_nan = [f for f in available if df[f].isna().all()]
    if all_nan:
        print(f"  ⚠  Dropping {len(all_nan)} all-NaN features: {all_nan[:4]}...")
        available = [f for f in available if f not in all_nan]

    # Impute NaN only for features that are MOSTLY present (>50% non-null).
    # Features that are mostly NaN (like PBP when merge failed) should be
    # dropped entirely — median-imputing a constant into 1000+ rows adds noise.
    mostly_nan = [f for f in available if df[f].isna().mean() > 0.5]
    if mostly_nan:
        print(f"  ⚠  Dropping {len(mostly_nan)} mostly-NaN features (>50% missing): {mostly_nan[:4]}")
        available = [f for f in available if f not in mostly_nan]

    for col in available:
        if df[col].isna().any():
            med = df[col].median()
            if not pd.isna(med):
                df[col] = df[col].fillna(med)

    df = df.dropna(subset=available + [TARGET_PTS, TARGET_TS]).copy()
    df = df[df.groupby("player")["player"].transform("count") >= 4]
    df = df.sort_values("game_date").reset_index(drop=True)

    if len(df) < 50:
        print(f"  [{label}] Not enough rows ({len(df)}) to train. Skipping.")
        return None

    cutoff = int(len(df) * 0.80)
    train, test = df.iloc[:cutoff], df.iloc[cutoff:]

    print(f"\n{'─'*52}")
    print(f"  Feature set: {label}")
    print(f"  Rows: {len(df)}  |  Train: {len(train)}  |  Test: {len(test)}")
    print(f"  Train up to: {train['game_date'].max().date()}")
    print(f"{'─'*52}\n")

    X_train, X_test = train[available], test[available]

    # Points regressor
    reg_pts = GradientBoostingRegressor(
        n_estimators=200, max_depth=2, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    reg_pts.fit(X_train, train[TARGET_PTS])
    pred_pts  = reg_pts.predict(X_test)
    base_pts  = test.groupby("player")[TARGET_PTS].transform(
        lambda s: train[train["player"].isin([s.name])][TARGET_PTS].mean()
        if s.name in train["player"].values else train[TARGET_PTS].mean()
    )
    mae_model    = mean_absolute_error(test[TARGET_PTS], pred_pts)
    mae_baseline = mean_absolute_error(test[TARGET_PTS], base_pts)
    r2           = r2_score(test[TARGET_PTS], pred_pts)

    print(f"  pts_per40 model:")
    print(f"    MAE:           {mae_model:.2f} pts/40")
    print(f"    Baseline MAE:  {mae_baseline:.2f} pts/40  (predict season avg)")
    print(f"    R²:            {r2:.3f}")
    diff = mae_baseline - mae_model
    if diff > 0:
        print(f"    Model is BETTER than baseline by {diff:.2f} pts/40")
    else:
        print(f"    Model is WORSE than baseline by {abs(diff):.2f} pts/40")

    # TS% regressor
    reg_ts   = GradientBoostingRegressor(
        n_estimators=200, max_depth=2, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    reg_ts.fit(X_train, train[TARGET_TS])
    pred_ts  = reg_ts.predict(X_test)
    mae_ts   = mean_absolute_error(test[TARGET_TS], pred_ts)
    r2_ts    = r2_score(test[TARGET_TS], pred_ts)
    print(f"\n  ts_pct model:")
    print(f"    MAE:  {mae_ts:.3f}  (±{mae_ts*100:.1f} pct points)")
    print(f"    R²:   {r2_ts:.3f}")

    # Classifier
    train_cls = train.copy()
    train_cls["above_avg"] = (
        train_cls[TARGET_PTS] > train_cls.groupby("player")[TARGET_PTS]
                                         .transform("mean")
    ).astype(int)
    test_cls  = test.copy()
    test_cls["above_avg"] = (
        test_cls[TARGET_PTS] > test_cls["player"].map(
            train_cls.groupby("player")[TARGET_PTS].mean()
        ).fillna(train_cls[TARGET_PTS].mean())
    ).astype(int)

    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=2, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    clf.fit(X_train, train_cls["above_avg"])
    pred_cls   = clf.predict(X_test)
    acc        = accuracy_score(test_cls["above_avg"], pred_cls)
    maj_class  = test_cls["above_avg"].value_counts(normalize=True).max()
    print(f"\n  above-avg classifier:")
    print(f"    Accuracy:  {acc*100:.1f}%")
    print(f"    Baseline:  {maj_class*100:.1f}%  (majority class)")

    # Feature importance
    imp = sorted(zip(available, reg_pts.feature_importances_),
                 key=lambda x: x[1], reverse=True)
    print(f"\n  Feature importance (pts_per40 model):")
    for feat, val in imp:
        bar = "█" * int(round(val * 40))
        print(f"    {feat:<28} {val*100:5.1f}%  {bar}")

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
        "train_cutoff": str(train["game_date"].max().date()),
        "pred_pts":  pred_pts,
        "pred_ts":   pred_ts,
        "pred_cls":  pred_cls,
        "test_df":   test,
        "reg_pts":   reg_pts,
    }


# ── Run models ────────────────────────────────────────────────────────────────
base_result = run_models(ncat.copy(), FEATURES_BASE, "BASE (no opponent rating)")

available_full = [f for f in FEATURES_FULL if f in ncat.columns]
missing_full   = [f for f in FEATURES_FULL if f not in ncat.columns]
if missing_full:
    print(f"\n  ⚠  Dropping {len(missing_full)} features not in data: {missing_full[:4]}")

# Add PBP features only if they were successfully merged (>50% non-null)
# Avoids the noise-imputation problem that hurt the previous run
pbp_available = [f for f in PBP_FEATURES
                 if f in ncat.columns and ncat[f].notna().mean() > 0.5]
if pbp_available:
    available_full = available_full + pbp_available
    print(f"  ✓  Adding {len(pbp_available)} PBP features: {pbp_available}")
else:
    print(f"  ⚠  PBP features unavailable — using opponent features only ({len(available_full)} features total)")

# Build FULL dataset — impute only features that are mostly present
ncat_full = ncat.copy()
for col in available_full:
    if col in ncat_full.columns and ncat_full[col].isna().any():
        if ncat_full[col].notna().mean() > 0.5:   # only impute if mostly present
            m = ncat_full[col].median()
            if not pd.isna(m):
                ncat_full[col] = ncat_full[col].fillna(m)
        else:
            # Mostly NaN — drop from feature set
            available_full = [f for f in available_full if f != col]
            print(f"  ⚠  Dropped {col} (mostly NaN)")

ncat_with_opp = ncat_full[ncat_full["opp_drtg_pregame"].notna()].copy()
full_result   = run_models(ncat_with_opp, available_full,
                           "FULL (with opponent difficulty)")


# ── Per-player season summary ─────────────────────────────────────────────────
print(f"\n{'─'*52}")
print(f"  Per-player season summary (women's)")
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


# ── Per-player prediction accuracy ───────────────────────────────────────────
best_result = full_result if full_result else base_result
if best_result:
    test_df  = best_result["test_df"]
    pred_pts = best_result["pred_pts"]
    model_label = best_result["label"]

    test_df = test_df.copy()
    test_df["predicted"] = pred_pts
    test_df["error"]     = abs(test_df["predicted"] - test_df[TARGET_PTS])

    acc_rows = []
    print(f"\n{'─'*52}")
    print(f"  Per-player prediction accuracy  (women's test set)")
    print(f"{'─'*52}")
    print(f"  (using {model_label})\n")
    print(f"  {'player':<25} {'test_games':>10} {'actual':>8} "
          f"{'predicted':>10} {'MAE':>6} {'bias':>7}")
    print(f"  {'─'*25} {'─'*10} {'─'*8} {'─'*10} {'─'*6} {'─'*7}")

    for player_name, grp in test_df.groupby("player"):
        if len(grp) < 3:
            continue
        actual    = grp[TARGET_PTS].mean()
        predicted = grp["predicted"].mean()
        mae       = grp["error"].mean()
        bias      = predicted - actual
        acc_rows.append({
            "player":     player_name,
            "test_games": len(grp),
            "actual_avg": round(float(actual),    2),
            "predicted_avg": round(float(predicted), 2),
            "mae":        round(float(mae),       2),
            "bias":       round(float(bias),      2),
        })
        bias_str = f"+{bias:5.2f}" if bias >= 0 else f"{bias:6.2f}"
        flag = "  ← model struggles" if mae > 6 else ""
        print(f"  {player_name:<25} {len(grp):>10}   {actual:>6.2f}     {predicted:>8.2f} "
              f"{mae:>6.2f} {bias_str}{flag}")

    acc_rows.sort(key=lambda r: r["mae"])
    print(f"\n  Most accurate:  {acc_rows[0]['player']}  (MAE {acc_rows[0]['mae']})")
    print(f"  Least accurate: {acc_rows[-1]['player']} (MAE {acc_rows[-1]['mae']})")

    accuracy_export = {
        "model_label": model_label,
        "per_player":  acc_rows,
    }
    acc_path = os.path.join(OUT_DIR, "player_prediction_accuracy_womens.json")
    with open(acc_path, "w") as f:
        json.dump(accuracy_export, f, indent=2)
    print(f"\n✅  player_prediction_accuracy_womens.json → {OUT_DIR}/")


# ── Export main model results JSON ────────────────────────────────────────────
use_result = full_result if full_result else base_result

# For women's data, BASE sometimes outperforms FULL due to smaller sample size
# and higher roster turnover — use whichever model has better R²
if full_result and base_result:
    if base_result["r2_pts"] > full_result["r2_pts"]:
        print(f"\n  ℹ  BASE model (R² {base_result['r2_pts']:.3f}) outperforms "
              f"FULL (R² {full_result['r2_pts']:.3f}) — using BASE as primary")
        use_result = base_result
    else:
        print(f"\n  ℹ  FULL model (R² {full_result['r2_pts']:.3f}) used as primary")
if not use_result:
    raise SystemExit("No model trained successfully.")

player_summary_records = summary.reset_index().to_dict("records")

model_export = {
    "base_model": {k: v for k, v in base_result.items()
                   if k not in ("pred_pts","pred_ts","pred_cls","test_df","reg_pts")}
    if base_result else None,
    "full_model": {k: v for k, v in use_result.items()
                   if k not in ("pred_pts","pred_ts","pred_cls","test_df","reg_pts")},
    "player_summary": player_summary_records,
    "gender": "womens",
}

results_path = os.path.join(OUT_DIR, "player_model_results_womens.json")
with open(results_path, "w") as f:
    json.dump(model_export, f, indent=2, default=str)

print(f"✅  player_model_results_womens.json  → {OUT_DIR}/")
print(f"✅  player_predictions_womens.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-STAT PROJECTION MODELS
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*60}")
print(f"  MULTI-STAT PROJECTION MODELS (women's)")
print(f"{'═'*60}")

STAT_TARGETS = {
    "pts_per40": "Points per 40",
    "reb_per40": "Rebounds per 40",
    "ast_per40": "Assists per 40",
    "to_per40":  "Turnovers per 40",
    "stl_per40": "Steals per 40",
    "blk_per40": "Blocks per 40",
}

multi_features = [f for f in available_full if f in ncat.columns]
multi_source   = ncat.copy()

# Impute NaN for opponent + PBP features before dropna.
# If a column is ALL NaN (e.g. PBP not yet scraped), remove it from
# multi_features entirely rather than trying to impute — dropna would
# otherwise drop every row.
impute_cols = ["opp_drtg_pregame", "opp_pace_pregame", "torvik_barthag",
               "torvik_adjoe", "rim_rate_r5", "three_rate_r5",
               "avg_shot_dist_r5", "assisted_rate_r5", "foul_drawn_rate_r5",
               "opp_rim_rate_r5", "opp_three_rate_r5"]
for col in impute_cols:
    if col not in multi_source.columns:
        multi_features = [f for f in multi_features if f != col]
        continue
    median_val = multi_source[col].median()
    if pd.isna(median_val):
        # All NaN — remove from feature set so dropna doesn't kill all rows
        multi_features = [f for f in multi_features if f != col]
        multi_source   = multi_source.drop(columns=[col], errors="ignore")
    else:
        multi_source[col] = multi_source[col].fillna(median_val)

multi_df = multi_source.dropna(
    subset=multi_features + list(STAT_TARGETS.keys())
).copy()
counts   = multi_df.groupby("player").size()
multi_df = multi_df[multi_df["player"].isin(counts[counts >= 4].index)]
multi_df = multi_df.sort_values("game_date").reset_index(drop=True)

cutoff = int(len(multi_df) * 0.80)
train  = multi_df.iloc[:cutoff]
test   = multi_df.iloc[cutoff:]

print(f"  Rows: {len(multi_df)}  |  Train: {len(train)}  |  Test: {len(test)}\n")

X_train, X_test = pd.DataFrame(train[multi_features]), pd.DataFrame(test[multi_features])

stat_models      = {}
stat_metrics     = {}
stat_predictions = {}

print(f"  {'Stat':<22} {'Model MAE':>10} {'Baseline':>10} {'R²':>7} {'Beats?':>8}")
print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*7} {'─'*8}")

for stat_col, stat_name in STAT_TARGETS.items():
    mdl = GradientBoostingRegressor(
        n_estimators=200, max_depth=2, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    mdl.fit(X_train, train[stat_col])
    preds = mdl.predict(X_test)
    player_avgs = train.groupby("player")[stat_col].mean().to_dict()
    test_baseline = test["player"].map(player_avgs).fillna(train[stat_col].mean())
    mae_model = mean_absolute_error(test[stat_col], preds)
    mae_base  = mean_absolute_error(test[stat_col], test_baseline)
    r2        = r2_score(test[stat_col], preds)
    stat_models[stat_col]  = mdl
    stat_metrics[stat_col] = {
        "stat_display":   stat_name,
        "mae_model":      round(float(mae_model), 3),
        "mae_baseline":   round(float(mae_base),  3),
        "r2":             round(float(r2),        3),
        "beats_baseline": bool(mae_model < mae_base),
    }
    stat_predictions[stat_col] = preds
    flag = "✓ YES" if mae_model < mae_base else "✗ no"
    print(f"  {stat_name:<22} {mae_model:>10.2f} {mae_base:>10.2f} {r2:>7.3f} {flag:>8}")

# Projection cards
print(f"\n  Generating per-player projection cards...")
latest = (multi_df.sort_values("game_date")
                  .groupby("player").tail(1)
                  .reset_index(drop=True))

projection_cards = []
for _, row in latest.iterrows():
    if row["player"] not in train["player"].values:
        continue
    player_feat = pd.DataFrame(row[multi_features].values.reshape(1, -1),
                               columns=multi_features)
    projections = {}
    for stat_col, mdl in stat_models.items():
        projections[stat_col] = round(float(mdl.predict(player_feat)[0]), 2)

    player_games = multi_df[multi_df["player"] == row["player"]]
    season_avgs  = {s: round(float(player_games[s].mean()), 2) for s in STAT_TARGETS}
    projection_cards.append({
        "player":        row["player"],
        "games_played":  int(len(player_games)),
        "avg_minutes":   round(float(player_games["min"].mean()), 1),
        "season_avg_ts_pct":  round(float(player_games["ts_pct"].mean()),  3),
        "season_avg_usg_pct": round(float(player_games["usg_pct"].mean()), 1),
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

projection_cards.sort(
    key=lambda c: c["projections"]["pts_per40"]["predicted"], reverse=True
)

multi_export = {
    "gender":          "womens",
    "feature_set":     "FULL" if "opp_drtg_pregame" in multi_features else "BASE",
    "train_rows":      int(len(train)),
    "test_rows":       int(len(test)),
    "train_cutoff":    str(train["game_date"].max().date()),
    "stat_metrics":    stat_metrics,
    "projection_cards": projection_cards,
}

multi_path = os.path.join(OUT_DIR, "player_multi_stat_projections_womens.json")
with open(multi_path, "w") as f:
    json.dump(multi_export, f, indent=2)

print(f"\n✅  player_multi_stat_projections_womens.json → {OUT_DIR}/")
print(f"    {len(projection_cards)} player projection cards exported")
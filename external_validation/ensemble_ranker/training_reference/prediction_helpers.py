from pathlib import Path
import numpy as np
import pandas as pd
import ranker_core as core

def load_test(path, train):
    frame = pd.read_csv(path, parse_dates=["date"], float_precision="round_trip")
    required = {"date", "ticker", "stock_name", "target1", *core.FIVE_FEATURES}
    if required - set(frame):
        raise ValueError(f"Missing test columns: {sorted(required - set(frame))}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Missing required test values")
    if not np.isfinite(frame[["target1", *core.FIVE_FEATURES]].to_numpy(float)).all():
        raise ValueError("Nonfinite test values")
    if frame.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate test date+ticker")
    overlap = sorted(set(frame.ticker) & set(train.ticker))
    if overlap:
        raise ValueError(f"Training/test ticker overlap: {overlap}")
    if frame.ticker.nunique() != 40 or train.ticker.nunique() != 66:
        raise ValueError("This experiment requires 66 training and 40 test tickers")
    calendar = train[["date", "week_index"]].drop_duplicates().sort_values("date")
    formal = calendar[calendar.week_index >= core.FORMAL_START_WEEK]
    included = frame.merge(formal, on="date", how="inner", validate="many_to_one")
    if set(included.date) != set(formal.date):
        raise ValueError("Incomplete test calendar")
    if included.groupby("date").size().min() < 5:
        raise ValueError("Fewer than five available stocks in a prediction week")
    included = included.sort_values(["week_index", "ticker"], kind="stable").reset_index(drop=True)
    included["target"] = included["target1"]
    included["target_verification_status"] = "user_confirmed_next_week_log_return_no_price_verification"
    for factor in core.FIVE_FEATURES:
        included[f"csrank__{factor}"] = included.groupby("week_index")[factor].rank(method="average", pct=True)
    included["actual_rank"] = included.groupby("week_index")["target"].rank(method="first", ascending=False).astype(int)
    audit = [
        ("test_input_rows", len(frame)), ("test_input_weeks", frame.date.nunique()),
        ("test_tickers", frame.ticker.nunique()), ("train_tickers", train.ticker.nunique()),
        ("ticker_overlap", len(overlap)), ("formal_test_rows", len(included)),
        ("formal_test_weeks", included.date.nunique()),
        ("excluded_rows_outside_formal_period", len(frame) - len(included)),
        ("minimum_formal_week_stocks", included.groupby("date").size().min()),
        ("maximum_formal_week_stocks", included.groupby("date").size().max()),
        ("missing_required_values", 0), ("duplicate_keys", 0),
        ("target2_used", False), ("test_targets_used_in_training", False),
    ]
    return included, pd.DataFrame(audit, columns=["check", "value"])


def training_window(train, spec, refit_week):
    part = train.loc[
        train.week_index.between(refit_week - spec.training_window_weeks, refit_week - 1)
        & train.label_end_week.le(refit_week)
    ].copy()
    if part.week_index.nunique() != spec.training_window_weeks:
        raise ValueError("Incomplete training window")
    if part.label_end_week.max() > refit_week or part.week_index.max() >= refit_week:
        raise AssertionError("Future label entered training")
    return part


def score_block(model, spec, block):
    # Only the explicitly named ranked factor columns enter predict().
    raw = model.predict(block[list(spec.ranked_features)].to_numpy(dtype=np.float32))
    output = block[["week_index", "date", "ticker", "target", "target_verification_status", "actual_rank"]].copy()
    output["raw_ranking_score"] = raw.astype(np.float64)
    output["percentile_score"] = output.groupby("week_index")["raw_ranking_score"].rank(method="average", pct=True)
    output["predicted_rank"] = output.groupby("week_index")["percentile_score"].rank(method="first", ascending=False).astype(int)
    return output


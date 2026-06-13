import pandas as pd

import research.validation as validation
from research.strategies.ma_strategy import Params


def test_stitched_walk_forward_equity_has_no_duplicate_dates(monkeypatch) -> None:
    index = pd.date_range("2015-01-01", periods=500, freq="D")
    data = pd.DataFrame({"TQQQ": 100.0, "QQQ": 100.0}, index=index)
    windows = [
        (index[0], index[260], index[261], index[300]),
        (index[0], index[260], index[300], index[340]),
    ]

    monkeypatch.setattr(
        validation,
        "walk_forward_windows",
        lambda dates, train_years, test_years, mode="rolling": windows,
    )
    monkeypatch.setattr(
        validation,
        "grid_search_on_window",
        lambda **kwargs: pd.DataFrame(
            [
                {
                    "short": 1,
                    "long": 2,
                    "ma_type": "sma",
                    "signal_asset": "TQQQ",
                    "threshold": 0.0,
                    "cooldown_days": 0,
                    "qqq_filter": 0,
                    "qqq_filter_window": 200,
                    "sharpe": 1.0,
                    "cagr": 0.1,
                    "max_dd": -0.1,
                    "excess_score": 0.2,
                }
            ]
        ),
    )

    def fake_make_eval_piece(
        full_data,
        params,
        eval_start,
        eval_end,
        transaction_cost_bps=10.0,
        prev_position=None,
        prev_risk_off_weight=None,
    ):
        piece_index = pd.date_range(eval_start, eval_end, freq="D")
        return pd.DataFrame(
            {
                "daily_ret": 0.0,
                "ret": 0.0,
                "position": 1.0,
                "turnover": 0.0,
            },
            index=piece_index,
        )

    monkeypatch.setattr(validation, "make_eval_piece", fake_make_eval_piece)
    monkeypatch.setattr(
        validation,
        "summarize_backtest",
        lambda bt, benchmark_ret, params: {
            "cagr": 0.1,
            "sharpe": 1.0,
            "max_dd": -0.1,
            "calmar": 1.0,
            "excess_cagr": 0.0,
            "excess_sharpe": 0.0,
            "excess_score": 0.0,
            "final_equity": 1.0,
        },
    )

    _, stitched = validation.walk_forward_search(
        data=data,
        grid=[Params(short=1, long=2, ma_type="sma")],
    )

    assert not stitched.index.has_duplicates

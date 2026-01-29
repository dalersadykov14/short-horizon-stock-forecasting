import unittest
import numpy as np
import pandas as pd

from backtest import walk_forward_backtest

class TestLeakageChecks(unittest.TestCase):
    def test_return_target_no_future_leakage(self):
        dates = pd.date_range("2020-01-01", periods=120, freq="B")
        prices = pd.Series(np.linspace(100, 200, num=120), index=dates)

        horizon = 5
        min_train = 20
        step = 5

        def model_fn(train):
            return 0.0

        preds, actuals, log = walk_forward_backtest(
            prices=prices,
            horizon=horizon,
            min_train=min_train,
            step=step,
            model_fn=model_fn,
            model_name="TEST",
            ticker="TEST",
            log_fn=lambda _: None,
            target_mode="return",
        )

        self.assertTrue(len(actuals) > 0)

        # Check first split target is based on train_end + horizon
        train_end_idx = min_train - 1
        target_idx = train_end_idx + horizon
        expected = float(np.log(prices.iloc[target_idx] / prices.iloc[train_end_idx]))
        self.assertAlmostEqual(actuals[0], expected, places=12)

        # Ensure target date strictly after train end date for all splits
        for rec in log:
            self.assertLess(rec["train_end_date"], rec["target_date"])

if __name__ == "__main__":
    unittest.main()

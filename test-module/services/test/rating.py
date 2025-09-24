import empyrical as ep
import pandas as pd


class Rating:
    def __init__(self):
        self.portfolio_values = {}

    def rate(self,portfolio_values):


        returns = pd.Series(portfolio_values).pct_change().dropna()

        metrics = {
            "total_return": float(ep.cum_returns_final(returns)),
            "cagr": float(ep.cagr(returns)),
            "sharpe_ratio": float(ep.sharpe_ratio(returns)),
            "sortino_ratio": float(ep.sortino_ratio(returns)),
            "max_drawdown": float(ep.max_drawdown(returns)),
            "calmar_ratio": float(ep.calmar_ratio(returns))
        }

        metrics["score"] = self.compute_score(metrics)

        return metrics


    def compute_score(self,metrics):
        """
        Przelicza metryki na wynik 0-100%
        """
        # normalizacja (progi do tuningu)
        sharpe_score = min(max(metrics["sharpe_ratio"] / 2.0, 0), 1)  # 2.0+ = max
        sortino_score = min(max(metrics["sortino_ratio"] / 3.0, 0), 1)  # 3.0+ = max
        cagr_score = min(max(metrics["cagr"] * 5, 0), 1)  # 20% CAGR = max
        dd_score = 1 - min(abs(metrics["max_drawdown"]) / 0.5, 1)  # -50% drawdown = 0

        # średnia ważona
        final_score = (
            0.4 * sharpe_score +
            0.3 * sortino_score +
            0.2 * cagr_score +
            0.1 * dd_score
        )

        return round(final_score * 100, 2)  # wynik %
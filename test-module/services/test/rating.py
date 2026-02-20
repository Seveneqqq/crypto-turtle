import empyrical as ep
import pandas as pd

class Rating:
    def __init__(self):
        self.portfolio_values = {}

    def rate(self, portfolio_values, transactions):
        returns = pd.Series(portfolio_values).pct_change().dropna()

        # Ustalona liczba okresów w roku dla danych minutowych
        annualization = 525_600

        metrics = {
            "total_return": float(ep.cum_returns_final(returns)) * 100,
            "cagr": float(ep.cagr(returns, annualization=annualization)) * 100,
            "sharpe_ratio": float(ep.sharpe_ratio(returns, annualization=annualization)),
            "sortino_ratio": float(ep.sortino_ratio(returns, annualization=annualization)),
            "max_drawdown": float(ep.max_drawdown(returns)) * 100,
            "calmar_ratio": float(ep.calmar_ratio(returns, annualization=annualization)),
            "growth_since_start": ((portfolio_values[-1] / portfolio_values[0]) - 1) * 100,
            "max_portfolio_growth": ((max(portfolio_values) / portfolio_values[0]) - 1) * 100,
            "max_portfolio_drop": ((min(portfolio_values) / portfolio_values[0]) - 1) * 100,
            "transaction_count": transactions,
            "final_portfolio_value":portfolio_values[-1]
        }


        metrics["avg_transactions_per_day"] = metrics["transaction_count"] / 60

        # Zaokrąglenie wartości
        for key, value in metrics.items():
            if isinstance(value, (float, int)):
                metrics[key] = round(value, 4)

        metrics["score"] = self.compute_score(metrics)
        return metrics

    def compute_score(self, metrics):
        sharpe_score = min(max(metrics["sharpe_ratio"] / 2.0, 0), 1)
        sortino_score = min(max(metrics["sortino_ratio"] / 3.0, 0), 1)
        cagr_score = min(max((metrics["cagr"] / 100) * 5, 0), 1)  # przeliczone z procentów
        dd_score = 1 - min(abs(metrics["max_drawdown"] / 100) / 0.5, 1)

        final_score = (
            0.4 * sharpe_score +
            0.3 * sortino_score +
            0.2 * cagr_score +
            0.1 * dd_score
        )
        return round(final_score * 100, 2)

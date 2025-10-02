from dataclasses import dataclass

@dataclass
class WalletConfig:
    wallet_percentage_limit: float = 0.9
    mincost: float = 1.0
    stop_loss: float = 0.05
    max_wallet_percentage_crypto: float = 0.5
    fee: float = 0.001       # stałe
    slippage: float = 0.0    # stałe

class Wallet:
    def __init__(self, balance, symbols, config: WalletConfig):
        self.balance = float(balance)
        self.holdings = {s: 0.0 for s in symbols}
        self.last_buy_price = {s: 0.0 for s in symbols}
        self.config = config

    def _exec_price(self, side, market_price):
        if side == "buy":
            return market_price * (1 + self.config.slippage)
        elif side == "sell":
            return market_price * (1 - self.config.slippage)
        return market_price

    def execute_order(self, side, symbol, qty, market_price):
        if qty <= 0 or market_price <= 0:
            return {"executed": 0.0, "value": 0.0, "status": "noop"}

        exec_price = self._exec_price(side, market_price)

        if side == "buy":
            cost = qty * exec_price * (1 + self.config.fee)
            if cost > self.balance or cost < self.config.mincost:
                return {"executed": 0.0, "value": 0.0, "status": "rejected"}
            port_val = self.total_value({symbol: market_price})
            new_asset_val = self.holdings[symbol]*market_price + qty*market_price
            if port_val > 0 and (new_asset_val / port_val) > self.config.max_wallet_percentage_crypto:
                return {"executed": 0.0, "value": 0.0, "status": "rejected"}
            self.balance -= cost
            self.holdings[symbol] += qty
            self.last_buy_price[symbol] = exec_price
            return {"executed": qty, "value": cost, "status": "filled"}

        elif side == "sell":
            available = self.holdings.get(symbol, 0.0)
            executed = min(qty, available)
            proceeds = executed * exec_price * (1 - self.config.fee)
            self.holdings[symbol] -= executed
            self.balance += proceeds
            if self.holdings[symbol] <= 0:
                self.last_buy_price[symbol] = 0.0
            return {"executed": executed, "value": proceeds, "status": "filled"}

        return {"executed": 0.0, "value": 0.0, "status": "error"}

    def total_value(self, prices: dict):
        total = self.balance
        for s, p in prices.items():
            total += self.holdings.get(s, 0.0) * p
        return total

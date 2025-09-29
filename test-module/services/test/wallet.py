# wallet.py
from dataclasses import dataclass

@dataclass
class WalletConfig:
    wallet_percentage_limit: float = 0.9  # max fraction of cash usable per tx
    mincost: float = 1.0                  # minimal USD per tx
    slippage: float = 0.0                 # fraction (e.g. 0.001)
    stop_loss: float = 0.05               # fraction (5% default)
    fee: float = 0.001                    # fraction (0.1% default)
    max_wallet_percentage_crypto: float = 0.5  # max share of portfolio in one asset

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

    def can_execute_buy(self, symbol, qty, price):
        cost = qty * price * (1 + self.config.fee)
        return cost <= self.balance and (qty * price) >= self.config.mincost

    def execute_order(self, side, symbol, qty, market_price):
        """
        side: "buy" or "sell"
        qty: quantity of crypto units to buy/sell (non-negative)
        returns: dict with executed_qty, executed_value_usd, status, message
        """
        if qty <= 0:
            return {"executed": 0.0, "value": 0.0, "status": "noop", "msg": "zero qty"}

        if market_price <= 0:
            return {"executed": 0.0, "value": 0.0, "status": "error", "msg": "bad price"}

        exec_price = self._exec_price(side, market_price)

        if side == "buy":
            cost = qty * exec_price * (1 + self.config.fee)
            if cost > self.balance:
                return {"executed": 0.0, "value": 0.0, "status": "rejected", "msg": "insufficient balance"}
            if cost < self.config.mincost:
                return {"executed": 0.0, "value": 0.0, "status": "rejected", "msg": "below mincost"}
            # check max wallet share
            port_val = self.total_value({symbol: market_price for symbol in self.holdings})
            new_asset_value = self.holdings[symbol]*market_price + qty*market_price
            if port_val > 0 and (new_asset_value / port_val) > self.config.max_wallet_percentage_crypto:
                return {"executed": 0.0, "value": 0.0, "status": "rejected", "msg": "max asset share"}
            # execute
            self.balance -= cost
            self.holdings[symbol] += qty
            self.last_buy_price[symbol] = exec_price
            return {"executed": qty, "value": cost, "status": "filled", "msg": ""}

        elif side == "sell":
            available = self.holdings.get(symbol, 0.0)
            if available <= 0:
                return {"executed": 0.0, "value": 0.0, "status": "rejected", "msg": "no holdings"}
            executed = min(qty, available)
            proceeds = executed * exec_price * (1 - self.config.fee)
            self.holdings[symbol] -= executed
            self.balance += proceeds
            # if sold everything, reset last_buy_price
            if self.holdings[symbol] <= 0:
                self.last_buy_price[symbol] = 0.0
            return {"executed": executed, "value": proceeds, "status": "filled", "msg": ""}

        else:
            return {"executed": 0.0, "value": 0.0, "status": "error", "msg": "bad side"}

    def total_value(self, prices: dict):
        total = self.balance
        for s, p in prices.items():
            total += self.holdings.get(s, 0.0) * p
        return total

    def check_stop_loss(self, symbol, market_price):
        """
        If price <= last_buy_price * (1 - stop_loss) -> sell all
        returns: same structure as execute_order
        """
        last = self.last_buy_price.get(symbol, 0.0)
        if last <= 0:
            return {"executed": 0.0, "value": 0.0, "status": "noop", "msg": "no last buy"}
        if market_price <= last * (1 - self.config.stop_loss):
            qty = self.holdings.get(symbol, 0.0)
            return self.execute_order("sell", symbol, qty, market_price)
        return {"executed": 0.0, "value": 0.0, "status": "noop", "msg": "no sl hit"}




# class Wallet:
#     def __init__(self, balance, symbols, threshold_buy, threshold_sell, wallet_percentage_limit, mincost, slippage, stop_loss, fee, max_wallet_percentage_crypto):
#         self.balance = balance
#         self.holdings = {symbol: 0.0 for symbol in symbols}
#         self.threshold_buy = threshold_buy
#         self.threshold_sell = threshold_sell
#         self.wallet_percentage_limit = wallet_percentage_limit
#         self.mincost = mincost
#         self.slippage = slippage
#         self.stop_loss = stop_loss
#         self.fee = fee
#         self.max_wallet_percentage_crypto = max_wallet_percentage_crypto
#
#         self.last_buy_price= {symbol: 0.0 for symbol in symbols}
#     def execute(self, action, symbol, price, model_value):
#
#         executed = 0.0
#         maxcost_value = (self.balance / price) * self.wallet_percentage_limit
#         mincost_value = self.mincost / price
#
#         # normalizacja model_value: 0.6 -> mincost, 1.0 -> maxcost
#         # if model_value < float(0.6):
#         #     model_value = 0.6
#
#         if action == "buy":
#             amount = mincost_value + (model_value - self.threshold_buy) / (1.0 - self.threshold_buy) * (maxcost_value - mincost_value)
#             cost = (amount * price) *(1 + self.fee)
#             if cost <= self.balance:
#                 self.balance -= cost
#                 self.holdings[symbol] += amount
#                 executed = amount
#                 self.last_buy_price[symbol] = price
#             else:
#                 action = "hold"
#                 executed = 0.0
#
#         elif action == "sell":
#             if self.holdings[symbol] > 0:
#                 amount = mincost_value + (model_value - self.threshold_sell) / (1.0 - self.threshold_sell) * (maxcost_value - mincost_value)
#                 if amount <= self.holdings[symbol]:
#                     self.holdings[symbol] -= amount
#                     self.balance += (amount * price) * (1 - self.fee)
#                     executed = amount
#                     sl_executed = self.stop_loss_check(symbol, price)
#                     if sl_executed > 0:
#                         executed = sl_executed
#                         action = "sell"
#                 else:
#                     action = "hold"
#                     executed = 0.0
#             else:
#                 action = "hold"
#                 executed = 0.0
#
#
#
#
#         return action, executed
#
#     def total_value(self, prices):
#         total = self.balance
#         for symbol, price in prices.items():
#             total += self.holdings[symbol] * price
#         return total
#
#     def stop_loss_check(self, symbol, price):
#         executed = 0.0
#         if self.holdings[symbol] > 0 and self.last_buy_price[symbol] > 0:
#             if price <= self.last_buy_price[symbol] * (1 - self.stop_loss):
#                 # sprzedaj całość przy stop loss
#                 executed = self.holdings[symbol]
#                 self.balance += executed * price * (1 - self.fee)
#                 self.holdings[symbol] = 0.0
#                 self.last_buy_price[symbol] = 0.0
#         return executed
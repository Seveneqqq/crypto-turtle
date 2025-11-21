# wallet.py
from dataclasses import dataclass

@dataclass
class WalletConfig:
    wallet_percentage_limit: float = 0.9  # max fraction of cash usable per tx
    mincost: float = 1.0                  # minimal USD per tx
    slippage: int = 0                 # fraction (e.g. 0.001)
    stop_loss: float = 0.05               # fraction (5% default)
    fee: float = 0.001                    # fraction (0.1% default)
    max_wallet_percentage_crypto: float = 0.5  # max share of portfolio in one asset

class Wallet:
    def __init__(self, balance, symbols, config: WalletConfig):
        self.balance = float(balance)
        self.holdings = {s: 0.0 for s in symbols}
        self.last_buy_price = {s: 0.0 for s in symbols}
        self.config = config
        self.orders = []
        self.logs = []

    def execute_orders(self, t, current_prices):

        self.calculate_orders(t, current_prices)

        current_orders = [o for o in self.orders if o["timestamp"] == t - self.config.slippage]

        if not current_orders:
            return

        for order in current_orders:
            action = order["action"]
            symbol = order["symbol"]
            match action:
                case "buy":
                    self.buy(order, current_prices[symbol])
                case "hold":
                    self.hold(order)
                case "sell":
                    self.sell(order, current_prices[symbol])
            self.logs.append(order)
            self.orders.remove(order)


    def buy(self, order, current_price):
        order["slippage_price"] = current_price
        qty = order["qty"]
        symbol = order["symbol"]

        usd_value = qty * current_price
        transaction_value = usd_value * (1 + self.config.fee)

        if transaction_value > self.balance:
            order["status"] = "rejected"
            return
        else:
            self.holdings[symbol] += qty
            self.balance -= transaction_value
            order["status"] = "accepted"
            return



    def hold(self, order):
        order["status"] = "accepted"
        return

    def sell(self, order, current_price):
        order["slippage_price"] = current_price
        qty = order["qty"]
        symbol = order["symbol"]

        usd_value = qty * current_price
        transaction_value = usd_value * (1 - self.config.fee)

        if qty > self.holdings.get(symbol, 0.0):
            order["status"] = "rejected"
            return
        else:
            self.holdings[order["symbol"]] -= qty
            self.balance += transaction_value
            order["status"] = "accepted"
            return

    def calculate_orders(self, t, prices):

        current_orders = [o for o in self.orders if o["timestamp"] == t]

        if not current_orders:
            return

        for order in current_orders:
            action = order["action"]
            match action:
                case "buy":
                    self.calculate_quantity_buy(order, prices)
                case "hold":
                    pass
                case "sell":
                    if self.holdings[order["symbol"]] > 0:
                        self.calculate_quantity_sell(order)
                    else:
                        order["action"] = "hold"
                        order["value"] = 1.0
                        order["reason"] = "no_holdings"
        return


    def calculate_quantity_buy(self, order, prices):
        frac = order["fraction"]
        symbol = order["symbol"]
        price = order["price"]

        available_cash = self.balance * self.config.wallet_percentage_limit
        usd_amount = available_cash * frac

        if usd_amount < self.config.mincost:
            order["action"] = "hold"
            order["value"] = 1.0
            order["reason"] = "lower_than_mincost"
            return

        qty = usd_amount / price

        crypto_qty = self.holdings.get(symbol, 0.0)
        current_holding_value = crypto_qty * price

        portfolio_value = self.total_value(prices)

        # Limit procentowy portfela
        if portfolio_value > 0:
            new_holding_value = current_holding_value + qty * price
            current_crypto_percentage = new_holding_value / portfolio_value

            if current_crypto_percentage > self.config.max_wallet_percentage_crypto:
                max_allowed_holding_value = self.config.max_wallet_percentage_crypto * portfolio_value
                max_allowed_qty = (max_allowed_holding_value - current_holding_value) / price

                qty = max(0.0, min(qty, max_allowed_qty))

                if qty == 0.0:
                    order["action"] = "hold"
                    order["value"] = 1.0
                    order["reason"] = "limit_max_wallet_percentage_crypto"
                    return
                if max_allowed_qty * price < self.config.mincost:
                    order["action"] = "hold"
                    order["value"] = 1.0
                    order["reason"] = "lower_than_mincost"
                    return
        order["qty"] = qty
        return

    def calculate_quantity_sell(self, order):
        frac = order["fraction"]
        symbol = order["symbol"]
        price = order["price"]


        crypto_qty = self.holdings.get(symbol, 0.0)
        qty = crypto_qty * frac

        usd_value = qty * price
        if usd_value < self.config.mincost:
            order["action"] = "hold"
            order["qty"] = 0.0
            order["reason"] = "lower_than_mincost"
            return

        order["qty"] = qty
        return

    def total_value(self, prices: dict):
        total = self.balance
        for s, p in prices.items():
            total += self.holdings.get(s, 0.0) * p
        return total

    def total_holding_value(self, symbol, prices):
        price = prices.get(symbol, 0.0)
        crypto_qty = self.holdings[symbol]
        return round(crypto_qty * price,4)


    def total_holdings(self, prices: dict):
        per_symbol_values = []

        for s, p in prices.items():
            value = self.holdings.get(s, 0.0) * p
            per_symbol_values.append(round(value, 4))

        return per_symbol_values

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
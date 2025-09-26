class Wallet:
    def __init__(self, balance, symbols, threshold_buy, threshold_sell, wallet_percentage_limit, mincost, slippage, stop_loss, fee, max_wallet_percentage_crypto):
        self.balance = balance
        self.holdings = {symbol: 0.0 for symbol in symbols}
        self.threshold_buy = threshold_buy
        self.threshold_sell = threshold_sell
        self.wallet_percentage_limit = wallet_percentage_limit
        self.mincost = mincost
        self.slippage = slippage
        self.stop_loss = stop_loss
        self.fee = fee
        self.max_wallet_percentage_crypto = max_wallet_percentage_crypto

        self.last_buy_price= {symbol: 0.0 for symbol in symbols}
    def execute(self, action, symbol, price, model_value):

        executed = 0.0
        maxcost_value = (self.balance / price) * self.wallet_percentage_limit
        mincost_value = self.mincost / price

        # normalizacja model_value: 0.6 -> mincost, 1.0 -> maxcost
        # if model_value < float(0.6):
        #     model_value = 0.6

        if action == "buy":
            amount = mincost_value + (model_value - self.threshold_buy) / (1.0 - self.threshold_buy) * (maxcost_value - mincost_value)
            cost = (amount * price) *(1 + self.fee)
            if cost <= self.balance:
                self.balance -= cost
                self.holdings[symbol] += amount
                executed = amount
                self.last_buy_price[symbol] = price
            else:
                action = "hold"
                executed = 0.0

        elif action == "sell":
            if self.holdings[symbol] > 0:
                amount = mincost_value + (model_value - self.threshold_sell) / (1.0 - self.threshold_sell) * (maxcost_value - mincost_value)
                if amount <= self.holdings[symbol]:
                    self.holdings[symbol] -= amount
                    self.balance += (amount * price) * (1 - self.fee)
                    executed = amount
                    sl_executed = self.stop_loss_check(symbol, price)
                    if sl_executed > 0:
                        executed = sl_executed
                        action = "sell"
                else:
                    action = "hold"
                    executed = 0.0
            else:
                action = "hold"
                executed = 0.0




        return action, executed

    def total_value(self, prices):
        total = self.balance
        for symbol, price in prices.items():
            total += self.holdings[symbol] * price
        return total

    def stop_loss_check(self, symbol, price):
        executed = 0.0
        if self.holdings[symbol] > 0 and self.last_buy_price[symbol] > 0:
            if price <= self.last_buy_price[symbol] * (1 - self.stop_loss):
                # sprzedaj całość przy stop loss
                executed = self.holdings[symbol]
                self.balance += executed * price * (1 - self.fee)
                self.holdings[symbol] = 0.0
                self.last_buy_price[symbol] = 0.0
        return executed
class Wallet:
    def __init__(self, balance, symbols, threshold_buy, threshold_sell,wallet_percentage_limit, mincost):
        self.balance = balance
        self.holdings = {symbol: 0.0 for symbol in symbols}
        self.threshold_buy = threshold_buy
        self.threshold_sell = threshold_sell
        self.wallet_percentage_limit = wallet_percentage_limit
        self.mincost = mincost
    def execute(self, action, symbol, price, model_value):

        executed = 0.0
        maxcost_value = (self.balance / price) * self.wallet_percentage_limit
        mincost_value = self.mincost / price

        # normalizacja model_value: 0.6 -> mincost, 1.0 -> maxcost
        if model_value < float(0.6):
            model_value = 0.6

        if action == "buy":
            amount = mincost_value + (model_value - self.threshold_buy) / (1.0 - self.threshold_buy) * (maxcost_value - mincost_value)
            cost = amount * price
            if cost <= self.balance:
                self.balance -= cost
                self.holdings[symbol] += amount
                executed = amount
            else:
                action = "hold"
                executed = 0.0

        elif action == "sell":
            if self.holdings[symbol] > 0:
                amount = mincost_value + (model_value - self.threshold_sell) / (1.0 - self.threshold_sell) * (maxcost_value - mincost_value)
                if amount <= self.holdings[symbol]:
                    self.holdings[symbol] -= amount
                    self.balance += amount * price
                    executed = amount
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
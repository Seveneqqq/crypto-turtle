class Wallet:
    def __init__(self, balance, symbols):
        self.balance = balance
        self.holdings = {symbol: 0.0 for symbol in symbols}

    def execute(self, action, symbol, price, model_value):
        """
        action: 'buy', 'sell', 'hold'
        symbol: np. 'BTC'
        price: aktualna cena
        model_value: float w zakresie [0.6, 1.0] - pewność modelu
        """
        executed = 0.0
        maxcost = self.balance / price  /2 #maksymalna ilość krypto do kupienia (połowa balance)
        mincost = 1.0 / price  # minimalna ilość krypto za 1$

        # normalizacja model_value: 0.6 -> mincost, 1.0 -> maxcost
        if model_value < float(0.6):
            model_value = 0.6
        amount = mincost + (model_value - 0.6) / (1.0 - 0.6) * (maxcost - mincost)

        print(f"[Wallet.execute] action={action} symbol={symbol} price={price} model_value={model_value}")
        if action == "buy":
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
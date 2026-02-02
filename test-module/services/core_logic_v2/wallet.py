from .structures import WalletConfig, CoreConfig

class Wallet:
    def __init__(self, core_mode = "sim_ml", core_config: CoreConfig = None):
        self.balance = core_config.balance_start
        self.symbols = core_config.symbols
        self.core_mode = core_mode
        self.config = WalletConfig()
        if self.core_mode in ("sim_rl","sim_ml"):  self.config.set(core_config)

        self.holdings = {s: 0.0 for s in self.symbols}
        self.last_buy_price = {s: 0.0 for s in self.symbols}
        self.avg_buy_price = {s: 0.0 for s in self.symbols}

        self.transaction_count = 0
        self.orders = []
        self.logs = []
        self.current_prices_wallet = {}

        self.portfolio_values_history = []
        self.portfolio_holdings_history = []

        # Callback Counters
        self.buys_count = 0
        self.sells_count = 0
        self.transaction_count = 0


    def reset(self, balance):
        """
        Resetuje stan portfela do wartości początkowych,
        ponownie wykorzystując istniejące obiekty w pamięci.
        """

        self.balance = float(balance)

        self.transaction_count = 0
        self.buys_count = 0
        self.sells_count = 0

        self.config = WalletConfig()
        self.current_prices_wallet.clear()
        self.portfolio_values_history.clear()
        self.portfolio_holdings_history.clear()
        self.orders.clear()
        self.logs.clear()

        for symbol in self.holdings:
            self.holdings[symbol] = 0.0
            self.last_buy_price[symbol] = 0.0
            self.avg_buy_price[symbol] = 0.0

    def update_history(self):
        """ Przygotowuje logi dla symulacji """
        total_value = self.total_value(self.current_prices_wallet)
        self.portfolio_values_history.append(round(total_value,1))
        self.portfolio_holdings_history.append(self.total_holdings(self.current_prices_wallet))

    def execute_orders(self, t, current_prices):
        self.current_prices_wallet = current_prices

        new_orders = [o for o in self.orders if o.timestamp == t]
        if not new_orders:
            pass
        else:
            for order in new_orders:
                if self.core_mode in ("rl_train", "sim_rl"): self.rl_config(order.rl_config)

                if order.action == "buy":  self.calculate_quantity_buy(order, current_prices)
                if order.action == "sell": self.calculate_quantity_sell(order)

        current_orders = [o for o in self.orders if o.timestamp == t - self.config.slippage]
        if not current_orders:
            return

        for order in current_orders:
            if order.status  == "rejected":
                pass
            else:
                if order.action == "buy":  self.buy(order)
                if order.action == "sell": self.sell(order)
            self.transaction_count += 1
            self.logs.append(order)
            self.orders.remove(order)

    def rl_config(self,scaled):
        pass
        # mincost_min, mincost_max = 1.0, 5.0
        # mincost = mincost_min + scaled[4] * (mincost_max - mincost_min)
        #
        # mwpc_min, mwpc_max = 0.7, 1.0
        # max_wallet_percentage_crypto = mwpc_min + scaled[5] * (mwpc_max - mwpc_min)
        #
        # self.config.mincost = float(round(mincost,2))
        # self.config.max_wallet_percentage_crypto = float(round(max_wallet_percentage_crypto,2))
        # self.config.stop_loss = scaled_dict[6]

    def buy(self, order):
        #TODO: Naprawić slippage
        current_price = order.price
        order.slippage_price = current_price
        qty = order.qty
        symbol = order.symbol

        usd_value = qty * current_price
        transaction_value = usd_value * (1 + self.config.fee)

        if transaction_value > self.balance:
            order.rejected("no_balance")
            return
        else:
            # --- LOGIKA ŚREDNIEJ WAŻONEJ CENY ZAKUPU ---
            current_qty = self.holdings.get(symbol, 0.0)
            current_avg = self.avg_buy_price.get(symbol, 0.0)

            # Jeśli już coś mamy, obliczamy nową średnią: (stara_wartość + nowa_wartość) / nowa_ilość
            if current_qty > 0:
                new_avg = ((current_qty * current_avg) + (qty * current_price)) / (current_qty + qty)
                self.avg_buy_price[symbol] = new_avg
            else:
                # Jeśli to pierwszy zakup
                self.avg_buy_price[symbol] = current_price
            # ---------------------------------------------
            self.holdings[symbol] += qty
            self.balance -= transaction_value
            self.buys_count +=1
            order.status = "accepted"
            return


    def sell(self, order):
        order.slippage_price = order.price

        symbol = order.symbol

        if order.qty > self.holdings.get(symbol, 0.0):
            order.rejected('not_enough_holdings')
            return
        else:
            usd_value = order.qty * order.price
            transaction_value = usd_value * (1 - self.config.fee)
            self.holdings[order.symbol] -= order.qty
            self.balance += transaction_value
            self.sells_count += 1
            order.status = "accepted"
            if self.holdings[symbol] <= 1e-9:
                self.holdings[symbol] = 0.0
                self.avg_buy_price[symbol] = 0.0
            return

    def calculate_quantity_buy(self, order, prices):

        price = order.price

        available_cash = self.balance * self.config.wallet_percentage_limit
        usd_amount = available_cash * order.fraction

        if usd_amount < self.config.mincost:
            order.rejected("lower_than_mincost")
            return

        qty = usd_amount / price

        crypto_qty = self.holdings.get(order.symbol, 0.0)
        current_holding_value = crypto_qty * price

        portfolio_value = self.total_value(prices)
        order.qty = qty
        # Limit procentowy portfela
        if portfolio_value > 0:
            new_holding_value = current_holding_value + qty * price
            current_crypto_percentage = new_holding_value / portfolio_value
            order.qty = qty
            if current_crypto_percentage > self.config.max_wallet_percentage_crypto:
                max_allowed_holding_value = self.config.max_wallet_percentage_crypto * portfolio_value
                max_allowed_qty = (max_allowed_holding_value - current_holding_value) / price

                qty = max(0.0, min(qty, max_allowed_qty))
                order.qty = qty
                if qty == 0.0:
                    order.rejected("limit_max_wallet_percentage_crypto")
                    return
                if max_allowed_qty * price < self.config.mincost:
                    order.rejected("lower_than_mincost")
                    return
        order.qty = qty
        return

    def calculate_quantity_sell(self, order):
        if self.holdings[order.symbol] > 0:
            crypto_qty = self.holdings.get(order.symbol, 0.0)
            qty = crypto_qty * order.fraction

            usd_value = qty * order.price
            if usd_value < self.config.mincost:
                order.rejected("lower_than_mincost")
                return
            order.qty = qty
        else: order.rejected("no_holdings")
        return

    def total_value(self, prices: dict):
        total = self.balance
        for s, p in prices.items():
            total += self.holdings.get(s, 0.0) * p
        #self.actual_portfolio_value = total
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
        # TODO: NOT IMPLEMENTED
        pass
        # last = self.last_buy_price.get(symbol, 0.0)
        # if last <= 0:
        #     return {"executed": 0.0, "value": 0.0, "status": "noop", "msg": "no last buy"}
        # if market_price <= last * (1 - self.config.stop_loss):
        #     qty = self.holdings.get(symbol, 0.0)
        #     return self.execute_order("sell", symbol, qty, market_price)
        # return {"executed": 0.0, "value": 0.0, "status": "noop", "msg": "no sl hit"}

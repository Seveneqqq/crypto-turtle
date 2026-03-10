from dataclasses import dataclass, asdict, field
from typing import Any, List, Optional
import numpy as np
import time


@dataclass(slots=True)
class Order:
    timestamp: int
    symbol: str
    action: str
    value: float
    buy_p: float = 0.0
    hold_p: float = 0.0
    sell_p: float = 0.0
    price: float = 0.0
    rl_config: Any = None
    fraction: float = 0.0
    qty: float = 0.0
    status: str = "created"
    reason: str = ""
    slippage_price: float = 0.0
    actual_qty: float = 0.0
    actual_usdt: float = 0.0
    total_portfolio_value: float = 0.0


    # Stan portfela PRZED transakcją
    local_before_balance: float = 0.0
    local_before_holdings: float = 0.0
    local_before_avg_buy_price: float = 0.0
    # Stan po lokalnej egzekucji (przed korektą Binance)
    local_after_balance: float = 0.0
    local_after_holdings: float = 0.0
    local_after_avg_buy_price: float = 0.0
    # Wynik Binance
    binance_status: str = ""
    binance_qty: float = 0.0
    binance_executed_price: float = 0.0
    binance_transaction_value: float = 0.0
    binance_order_id: str = ""
    binance_message: str = ""
    # Stan po korekcie (ground truth)
    reconciled_balance: float = 0.0
    reconciled_holdings: float = 0.0
    reconciled_avg_buy_price: float = 0.0
    # Drift local_after vs reconciled
    drift_qty: float = 0.0
    drift_value: float = 0.0


    # , "buy:0","hold":1, "sell:2"
    def rejected(self, reason: str):
        self.status = "rejected"
        self.reason = reason

    def to_dict(self):
        d = asdict(self)
        if hasattr(self.rl_config, "tolist"):
            d['rl_config'] = self.rl_config.tolist()
        return d

@dataclass
class PendingBinanceOrder:
    """
    Zlecenie wysłane do Binance czekające na potwierdzenie.
    Przechowywane w pamięci serwisu między iteracjami pętli.
    """
    order_id: str                       # binance orderId
    symbol: str
    action: str                         # "buy" | "sell"
    local_qty: float                    # qty odejęte/dodane lokalnie
    local_price: float
    local_balance_before: float         # snapshot do rollbacku
    local_holdings_before: float
    local_avg_buy_price_before: float
    submitted_at: float = field(default_factory=time.time)

    def age_seconds(self) -> float:
        return time.time() - self.submitted_at


@dataclass(slots=True)
class PreviousEpoch:
    epoch: int = 1

    buys: int = 0
    holds: int = 0
    sells: int = 0
    transactions: int = 0

    rewards: List[float] = field(default_factory=list)
    min_r: float = 0.0
    max_r: float = 0.0
    mean_r: float = 0.0
    end_r:float = 0.0

    portfolio_values: List[float] = field(default_factory=list)
    min_p: float = 0.0
    max_p: float = 0.0
    mean_p: float = 0.0
    end_p: float = 0.0

    # PO ostatnim kroku okresu danych (teraz ucinane do 10240) w zależności od okresu dla jednej iteracji
    def calculate_history(self):
        self.holds = self.transactions - self.sells - self.buys

        self.min_r = np.min(self.rewards)
        self.max_r = np.max(self.rewards)
        self.mean_r = round(np.mean(self.rewards),0)
        self.end_r = self.rewards[-1]
        self.rewards = self.rewards[::50]

        self.min_p = np.min(self.portfolio_values)
        self.max_p = np.max(self.portfolio_values)
        self.mean_p = round(np.mean(self.portfolio_values),0)
        self.end_p = self.portfolio_values[-1]
        self.portfolio_values = self.portfolio_values[::50]

    def to_dict(self):
        """Przygotowuje słownik bezpieczny dla JSON (rzutowanie typów)."""
        return {
            "epoch": int(self.epoch),
            "metrics": {
                "buys": int(self.buys),
                "sells": int(self.sells),
                "holds": int(self.holds)
            },
            "portfolio": {
                "portfolio_values": [float(x) for x in self.portfolio_values],
                "min_p": float(self.min_p),
                "mean_p": float(self.mean_p),
                "max_p": float(self.max_p),
            },
            "reward": {
                "reward_values": [float(x) for x in self.rewards],
                "min_r": float(self.min_r),
                "mean_r": float(self.mean_r),
                "max_r": float(self.max_r),
            }
        }
@dataclass(slots=True)
class CoreConfig:
    sim_mode: str = ""
    balance_start: float = 0.0
    add_fee: bool = True
    symbols: List[str] = field(default_factory=list)
    rlagent: str = ""
    rlagent_deterministic: bool = True
    slippage: int = 0
    wallet_percentage_limit: float = 0.0
    max_wallet_percentage_crypto: float = 0.0
    stop_loss: float = 0.0
    threshold_buy: float = 0.0
    threshold_sell: float = 0.0
    mincost: float = 0.0
    tx_fraction: float = 0.0
    max_fraction_when_confident: float = 0.0
    experiment_version: str = ""

    def to_dict(self):
        d = asdict(self)
        return d

@dataclass(slots=True)
class DecisionConfig:
    threshold_buy: float = 0.5
    threshold_sell: float = 0.5
    tx_fraction: float = 0.1
    max_fraction_when_confident: float = 0.5
    clamp_model_value: bool = True
    fraction_buy: float = 0.3
    fraction_sell: float = 0.3

    def set(self, output_config: CoreConfig):
        self.threshold_buy = output_config.threshold_buy
        self.threshold_sell = output_config.threshold_sell
        self.tx_fraction = output_config.tx_fraction
        self.max_fraction_when_confident = output_config.max_fraction_when_confident
        #print(output_config)

class WalletConfig:
    wallet_percentage_limit: float = 0.9
    mincost: float = 1.0
    slippage: int = 0
    stop_loss: float = 0.05
    fee: float = 0.001
    max_wallet_percentage_crypto: float = 1.0

    def set(self, output_config: CoreConfig):
        self.wallet_percentage_limit = output_config.wallet_percentage_limit
        self.mincost = output_config.mincost
        self.slippage = output_config.slippage
        self.stop_loss = output_config.stop_loss
        if not output_config.add_fee: self.fee = 0.0
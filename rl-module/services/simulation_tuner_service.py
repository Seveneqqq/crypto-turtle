from .rl.base_tuner_service import BaseTunerService
from module_loader import block_simulation
import optuna
import os


class SimulationTunerService(BaseTunerService):
    """Service for tuning simulation parameters using only ML models (no RL agent)."""

    def __init__(self, base_dir=None):
        super().__init__(base_dir=base_dir)

        self.agent_type = 'simulation'
        print(f"[SimulationTuner] Service ready.")

    def _objective(self, trial: optuna.Trial) -> float:
        """Simulation parameters objective"""

        # Simulation parameters to tune
        threshold_buy = self._get_param(trial, 'threshold_buy', 0.45)
        threshold_sell = self._get_param(trial, 'threshold_sell', 0.45)
        wallet_percentage_limit = self._get_param(trial, 'wallet_percentage_limit', 0.9)
        max_wallet_percentage_crypto = self._get_param(trial, 'max_wallet_percentage_crypto', 1.0)
        stop_loss = self._get_param(trial, 'stop_loss', 1.0)
        tx_fraction = self._get_param(trial, 'tx_fraction', 0.01)
        max_fraction_when_confident = self._get_param(trial, 'max_fraction_when_confident', 0.5)
        slippage = self._get_param(trial, 'slippage', 0)
        mincost = self._get_param(trial, 'mincost', 1.0)
        balance_start = self._get_param(trial, 'balance_start', 1000.0)

        # Print trial info
        print(f"\n{'=' * 70}")
        print(f"Trial {trial.number} - Simulation Parameters (ML Models Only)")
        print(f"{'=' * 70}")
        print(f"  threshold_buy: {threshold_buy:.3f}")
        print(f"  threshold_sell: {threshold_sell:.3f}")
        print(f"  wallet_percentage_limit: {wallet_percentage_limit:.3f}")
        print(f"  max_wallet_percentage_crypto: {max_wallet_percentage_crypto:.3f}")
        print(f"  stop_loss: {stop_loss:.3f}")
        print(f"  tx_fraction: {tx_fraction:.4f}")
        print(f"  max_fraction_when_confident: {max_fraction_when_confident:.3f}")
        print(f"  slippage: {slippage:.4f}")
        print(f"  mincost: {mincost:.2f}")
        print(f"  balance_start: ${balance_start:.2f}")
        print(f"{'=' * 70}\n")

        try:
            # Run evaluation with these simulation parameters
            return self._evaluate_ml_simulation(
                trial=trial,
                threshold_buy=threshold_buy,
                threshold_sell=threshold_sell,
                wallet_percentage_limit=wallet_percentage_limit,
                max_wallet_percentage_crypto=max_wallet_percentage_crypto,
                stop_loss=stop_loss,
                tx_fraction=tx_fraction,
                max_fraction_when_confident=max_fraction_when_confident,
                slippage=slippage,
                mincost=mincost,
                balance_start=balance_start
            )

        except Exception as e:
            print(f"\n[Simulation Trial {trial.number}] FAILED: {e}")
            import traceback
            traceback.print_exc()
            return -1000.0




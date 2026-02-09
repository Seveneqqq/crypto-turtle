"""
Optuna Hyperparameter Tuning for PPO Trading Agent - FIXED VERSION

Changes:
- Extracted hyperparameter suggestions to helper methods
- Allows subclassing without dynamic value space errors
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import os
import json
from datetime import datetime
from typing import Dict, Any

from module_loader import block_train_ppo, block_simulation


class OptunaPPOTuner:
    def __init__(self,
                 model_files: Dict[str, str],
                 train_data_files: Dict[str, str],
                 eval_data_files: Dict[str, str],
                 study_name: str = None,
                 storage: str = None):

        self.model_files = model_files
        self.train_data_files = train_data_files
        self.eval_data_files = eval_data_files

        if study_name is None:
            study_name = f"ppo_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if storage is None:
            storage = f"sqlite:///optuna_studies/{study_name}.db"
            os.makedirs("optuna_studies", exist_ok=True)

        self.study_name = study_name
        self.storage = storage

        self.results_file = f"optuna_studies/{study_name}_results.json"
        self.best_params_file = f"optuna_studies/{study_name}_best_params.json"

    # === HELPER METHODS FOR HYPERPARAMETERS ===
    # Subclass can override these to change search space

    def get_episode_timesteps(self, trial):
        """Episode length to tune"""
        return trial.suggest_categorical('episode_timesteps', [256, 512, 1024, 2048])

    def get_learning_rate(self, trial):
        """Learning rate range"""
        return trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)

    def get_n_steps_multiplier(self, trial):
        """n_steps = episode_timesteps × multiplier"""
        return trial.suggest_categorical('n_steps_multiplier', [1, 2, 4])

    def get_batch_size(self, trial):
        """Batch size options"""
        return trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])

    def get_gamma(self, trial):
        """Discount factor"""
        return trial.suggest_float('gamma', 0.95, 0.9999, log=True)

    def get_ent_coef(self, trial):
        """Entropy coefficient"""
        return trial.suggest_float('ent_coef', 0.0001, 0.1, log=True)

    def get_ml_threshold(self, trial):
        """ML model threshold"""
        return trial.suggest_float('ml_threshold', 0.35, 0.60)

    def get_total_timesteps(self, trial):
        """Total training timesteps"""
        return trial.suggest_categorical('total_timesteps', [500_000, 1_000_000, 2_000_000])

    # === MAIN OBJECTIVE FUNCTION ===

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective - returns metric to MAXIMIZE"""

        # Get hyperparameters using helper methods
        episode_timesteps = self.get_episode_timesteps(trial)
        learning_rate = self.get_learning_rate(trial)
        n_steps_multiplier = self.get_n_steps_multiplier(trial)
        batch_size = self.get_batch_size(trial)
        gamma = self.get_gamma(trial)
        ent_coef = self.get_ent_coef(trial)
        ml_threshold = self.get_ml_threshold(trial)
        total_timesteps = self.get_total_timesteps(trial)

        # Calculate n_steps
        n_steps = episode_timesteps * n_steps_multiplier

        # Ensure batch_size <= n_steps
        if batch_size > n_steps:
            batch_size = n_steps // 2

        print(f"\n{'='*70}")
        print(f"Trial {trial.number}")
        print(f"{'='*70}")
        print(f"  episode_timesteps: {episode_timesteps}")
        print(f"  n_steps: {n_steps} (episode × {n_steps_multiplier})")
        print(f"  learning_rate: {learning_rate:.6f}")
        print(f"  batch_size: {batch_size}")
        print(f"  gamma: {gamma:.4f}")
        print(f"  ent_coef: {ent_coef:.6f}")
        print(f"  ml_threshold: {ml_threshold:.2f}")
        print(f"  total_timesteps: {total_timesteps:,}")
        print(f"{'='*70}\n")

        try:
            # Set environment variables for training
            os.environ['EPISODE_TIMESTEPS'] = str(episode_timesteps)
            os.environ['ML_THRESHOLD'] = str(ml_threshold)

            # Train agent
            agent_result = block_train_ppo().train_agent_ppo(
                rabbitmq_msg_id=None,
                model_files=self.model_files,
                data_files=self.train_data_files,
                total_timesteps=total_timesteps,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                gamma=gamma,
                ent_coef=ent_coef
            )

            if agent_result["result"] != "success":
                raise RuntimeError(f"Training failed: {agent_result['message']}")

            agent_path = agent_result["agent_file"]

            # Evaluate on test set
            sim_result = block_simulation().run_simulation(
                rabbit_msg_id=None,
                model_paths=self.model_files,
                data_files=self.eval_data_files,
                agent_rl=agent_path,
                balance_start=1000.0,
                add_fee=True,
                sim_mode="rl",
                rlagent_deterministic=True,
                slippage=0,
                wallet_percentage_limit=0.9,
                max_wallet_percentage_crypto=1.0,
                stop_loss=1.0,
                threshold_buy=ml_threshold,
                threshold_sell=ml_threshold,
                mincost=1.0,
                tx_fraction=0.01,
                max_fraction_when_confident=0.5,
                plots=False,
                logs=False,
                results=True
            )

            # Extract metrics
            final_value = sim_result.get('final_portfolio_value', 1000.0)
            sharpe = sim_result.get('sharpe_ratio', 0.0)
            max_dd = sim_result.get('max_drawdown', -100.0)

            # Metric to maximize: percentage return
            metric = (final_value - 1000.0) / 1000.0 * 100

            print(f"\nTrial {trial.number} Result:")
            print(f"  Final Value: ${final_value:.2f}")
            print(f"  Return: {metric:.2f}%")
            print(f"  Sharpe: {sharpe:.2f}")
            print(f"  Max DD: {max_dd:.2f}%\n")

            # Log user attributes
            trial.set_user_attr('final_value', final_value)
            trial.set_user_attr('sharpe_ratio', sharpe)
            trial.set_user_attr('max_drawdown', max_dd)
            trial.set_user_attr('agent_file', agent_path)

            # Save trial result
            self._save_trial_result(trial, metric, final_value, sharpe, max_dd, agent_path)

            return metric

        except Exception as e:
            print(f"\nTrial {trial.number} FAILED: {e}")
            import traceback
            traceback.print_exc()
            return -1000.0

    def _save_trial_result(self, trial, metric, final_value, sharpe, max_dd, agent_path):
        """Save individual trial result to JSON"""
        result = {
            'trial_number': trial.number,
            'params': trial.params,
            'metric': metric,
            'final_value': final_value,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'agent_file': agent_path,
            'timestamp': datetime.now().isoformat()
        }

        results = []
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                results = json.load(f)

        results.append(result)

        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2)

    def run_study(self, n_trials: int = 50, n_jobs: int = 1, timeout: int = None):
        """Run Optuna study"""

        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True,
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )

        print(f"\nStarting Optuna Study: {self.study_name}")
        print(f"Trials: {n_trials}, Parallel Jobs: {n_jobs}")
        print(f"Storage: {self.storage}\n")

        study.optimize(
            self.objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            timeout=timeout,
            show_progress_bar=True
        )

        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETE")
        print("="*70)

        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best metric: {study.best_value:.2f}%")
        print(f"\nBest parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        # Save best parameters
        best_params = {
            'best_trial': study.best_trial.number,
            'best_metric': study.best_value,
            'best_params': study.best_params,
            'best_agent': study.best_trial.user_attrs.get('agent_file'),
            'timestamp': datetime.now().isoformat()
        }

        with open(self.best_params_file, 'w') as f:
            json.dump(best_params, f, indent=2)

        print(f"\nBest parameters saved to: {self.best_params_file}")
        print(f"All results saved to: {self.results_file}")

        self._generate_report(study)

        return study

    def _generate_report(self, study):
        """Generate optimization report"""
        trials_df = study.trials_dataframe()

        report_file = f"optuna_studies/{self.study_name}_report.csv"
        trials_df.to_csv(report_file, index=False)

        print(f"Full report saved to: {report_file}")

        print("\nTop 5 trials:")
        top_trials = trials_df.nlargest(5, 'value')
        print(top_trials[['number', 'value', 'params_episode_timesteps',
                          'params_learning_rate', 'params_n_steps_multiplier']])


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Optuna PPO Hyperparameter Tuning')
    parser.add_argument('--n-trials', type=int, default=50, help='Number of trials')
    parser.add_argument('--n-jobs', type=int, default=1, help='Parallel jobs')
    parser.add_argument('--timeout', type=int, default=None, help='Timeout in seconds')
    parser.add_argument('--study-name', type=str, default=None, help='Study name')

    args = parser.parse_args()

    # Configuration
    model_files = {
        "BTC": "btc_model_c_v1_1210_11.pkl"
    }

    train_data_files = {
        "BTC": "processed_ROBUST_set4_BTCUSDT-agent_1m_5m_2025-10-11_2d07d1b6-baf3-4ed8-bdc0-390156e92dc6.csv"
    }

    eval_data_files = {
        "BTC": "processed_ROBUST_set4_BTCUSDT-simulation_1m_2m_2025-10-11_8a076fe5-6d3c-4b1c-a490-9be3a3174517.csv"
    }

    tuner = OptunaPPOTuner(
        model_files=model_files,
        train_data_files=train_data_files,
        eval_data_files=eval_data_files,
        study_name=args.study_name
    )

    study = tuner.run_study(
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        timeout=args.timeout
    )


if __name__ == "__main__":
    main()
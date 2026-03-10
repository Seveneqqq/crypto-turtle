import os
import json
from datetime import datetime
from typing import Dict, List, Any
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from z_maina_funkcje.module_loader import block_simulation

class BaseTunerService:

    def __init__(self, base_dir: str = None):

        self.base_dir = base_dir
        self.model_files = {}
        self.train_data_files = {}
        self.eval_data_files = {}
        self.log_path = os.path.join(self.base_dir, "shared", "agents", "optuna_studies")

        self.sim = block_simulation()

        self.agent_type = None
        self.live_logging = False
        self.experiment_version = None
        self.config = None
        self.defaults = None
        self.study_name = None
        self.storage = None
        self.results_file = None
        self.best_params_file = None

    def _parse_config(self, config: Dict[str, List], defaults: Dict[str, Any] = None):
        """Dynamiczne tworzenie funkcji dla parametrów"""
        self.config = config
        self.defaults = defaults or {}

        # Create getters for parameters in config
        for param_name, spec in config.items():
            param_type = spec[0]
            values = spec[1:]

            if param_type == "C":
                # Categorical or Fixed
                if len(values) == 1:
                    # Fixed value
                    def getter(trial, val=values[0]):
                        return val
                else:
                    # Categorical
                    def getter(trial, name=param_name, choices=values):
                        return trial.suggest_categorical(name, choices)

            elif param_type == "F":
                # Float (linear scale)
                low, high = values[0], values[1]

                def getter(trial, name=param_name, l=low, h=high):
                    return trial.suggest_float(name, l, h, log=False)

            elif param_type == "FL":
                # Float (log scale)
                low, high = values[0], values[1]

                def getter(trial, name=param_name, l=low, h=high):
                    return trial.suggest_float(name, l, h, log=True)

            elif param_type == "I":
                # Integer
                low, high = values[0], values[1]

                def getter(trial, name=param_name, l=low, h=high):
                    return trial.suggest_int(name, l, h)

            else:
                raise ValueError(f"Nieznany typ parametru: {param_type} dla parametru: {param_name}")

            # Dynamically set getter method
            setattr(self, f"get_{param_name}", getter)

        # Create static getters for defaults
        for param_name, default_value in self.defaults.items():
            def getter(trial, val=default_value):
                return val

            setattr(self, f"get_{param_name}", getter)

    def _get_param(self, trial: optuna.Trial, param_name: str, fallback_default: Any = None):
        """Wyciąganie parametrów ze słownika"""
        getter_name = f"get_{param_name}"
        if hasattr(self, getter_name):
            return getattr(self, getter_name)(trial)
        elif fallback_default is not None:
            return fallback_default
        else:
            raise ValueError(f"Nie ma takiego parametru: {param_name}")

    def _evaluate_agent(self,
                        trial: optuna.Trial,
                        agent_filename: str,
                        ml_threshold: float) -> float:
        """Ewaluacja na zbiorze testowym."""
        try:
            print(f"[{self.agent_type.upper()} Trial {trial.number}] Ewaluacja...")

            sim_result = block_simulation().run_simulation(
                rabbit_msg_id=None,
                model_paths=self.model_files,
                data_files=self.eval_data_files,
                agent_rl=agent_filename,
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
                results=True,
                experiment_version=self.experiment_version
            )

            final_value = sim_result.get('final_portfolio_value', 1000.0)
            sharpe = sim_result.get('sharpe_ratio', 0.0)
            max_dd = sim_result.get('max_drawdown', -100.0)
            metric = (final_value - 1000.0) / 1000.0 * 100

            print(f"\n[{self.agent_type.upper()} Trial {trial.number}] - wyniki:")
            print(f"  Final Value: ${final_value:.2f}")
            print(f"  Return: {metric:.2f}%")
            print(f"  Sharpe: {sharpe:.2f}")
            print(f"  Max DD: {max_dd:.2f}%\n")

            trial.set_user_attr('final_value', final_value)
            trial.set_user_attr('sharpe_ratio', sharpe)
            trial.set_user_attr('max_drawdown', max_dd)
            trial.set_user_attr('agent_file', agent_filename)

            # Save trial result
            self._save_trial_result(trial, metric, final_value, sharpe, max_dd, agent_filename)

            return metric

        except Exception as e:
            print(f"\n[{self.agent_type.upper()} Trial {trial.number}] Evaluation FAILED: {e}")
            import traceback
            traceback.print_exc()
            return -1000.0

    def _evaluate_ml_simulation(self,
                                trial: optuna.Trial,
                                threshold_buy: float,
                                threshold_sell: float,
                                wallet_percentage_limit: float,
                                max_wallet_percentage_crypto: float,
                                stop_loss: float,
                                tx_fraction: float,
                                max_fraction_when_confident: float,
                                slippage: float,
                                mincost: float,
                                balance_start: float) -> float:
        """Evaluate simulation using only ML models (no RL agent)."""



        try:
            print(f"[Simulation Trial {trial.number}] Running ML-only evaluation...")

            sim_result = self.sim.run_simulation(
                rabbit_msg_id=None,
                model_paths=self.model_files,
                data_files=self.eval_data_files,
                agent_rl=None,  # NO RL AGENT - ML models only!
                balance_start=balance_start,
                add_fee=True,
                sim_mode="ml",  # ML mode instead of "rl"
                rlagent_deterministic=False,  # Not applicable for ML mode
                slippage=slippage,
                wallet_percentage_limit=wallet_percentage_limit,
                max_wallet_percentage_crypto=max_wallet_percentage_crypto,
                stop_loss=stop_loss,
                threshold_buy=threshold_buy,
                threshold_sell=threshold_sell,
                mincost=mincost,
                tx_fraction=tx_fraction,
                max_fraction_when_confident=max_fraction_when_confident,
                plots=False,
                logs=False,
                results=False,
                experiment_version=self.experiment_version
            )

            final_value = sim_result.get('final_portfolio_value', balance_start)
            sharpe = sim_result.get('sharpe_ratio', 0.0)
            max_dd = sim_result.get('max_drawdown', -100.0)

            # Calculate return percentage
            metric = (final_value - balance_start) / balance_start * 100

            print(f"\n[Simulation Trial {trial.number}] - Results:")
            print(f"  Final Value: ${final_value:.2f}")
            print(f"  Return: {metric:.2f}%")
            print(f"  Sharpe: {sharpe:.2f}")
            print(f"  Max DD: {max_dd:.2f}%\n")

            trial.set_user_attr('final_value', final_value)
            trial.set_user_attr('sharpe_ratio', sharpe)
            trial.set_user_attr('max_drawdown', max_dd)
            trial.set_user_attr('sim_mode', 'ml')

            # Save trial result (without agent_path since we don't use RL agent)
            self._save_trial_result(trial, metric, final_value, sharpe, max_dd, agent_path=None)

            return metric

        except Exception as e:
            print(f"\n[Simulation Trial {trial.number}] Evaluation FAILED: {e}")
            import traceback
            traceback.print_exc()
            return -1000.0

    def run_tuning(self,
                   model_files: Dict[str, str],
                   eval_data_files: Dict[str, str],
                   train_data_files: Dict[str, str] = None,
                   config: Dict[str, List] = None,
                   defaults: Dict[str, Any] = None,
                   n_trials: int = 50,
                   n_jobs: int = 1,
                   timeout: int = None,
                   study_name: str = None,
                   live_logging: bool = True,
                   experiment_version: str = "v0"
                   ) -> optuna.Study:
        """Run hyperparameter tuning."""

        # ZAPISUJEMY PLIKI DO INSTANCJI
        self.model_files = model_files
        self.train_data_files = train_data_files
        self.eval_data_files = eval_data_files
        self.live_logging = live_logging
        self.experiment_version = experiment_version
        # Use empty config if None
        if config is None:
            config = {}

        # Generate study name
        if study_name is None:
            timestamp = datetime.now().strftime("%d-%m__%H-%M-%S")
            study_name = f"{self.agent_type}_tuning_{timestamp}"

        self.study_name = study_name
        self.storage = f"sqlite:///{self.log_path}/{study_name}.db"
        os.makedirs(self.log_path, exist_ok=True)

        # Results files
        self.results_file = os.path.join(self.log_path,f"{study_name}_results.json")
        self.best_params_file = os.path.join(self.log_path,f"{study_name}_best_params.json")

        # Parse config and create getters
        self._parse_config(config, defaults)

        # Create study
        study = optuna.create_study(
            study_name=study_name,
            storage=self.storage,
            load_if_exists=True,
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )

        print(f"\n{'=' * 70}")
        print(f"Starting Optuna Study: {study_name}")
        print(f"{'=' * 70}")
        print(f"Algorithm: {self.agent_type.upper()}")
        print(f"Trials: {n_trials}")
        print(f"Parallel Jobs: {n_jobs}")
        print(f"Tuned params: {len(config)}")
        print(f"Static params: {len(defaults) if defaults else 0}")
        print(f"Storage: {self.storage}\n")

        # Run optimization
        study.optimize(
            self._objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            timeout=timeout,
            show_progress_bar=True
        )

        # Print results
        print(f"\n{'=' * 70}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'=' * 70}")
        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best metric: {study.best_value:.2f}%")
        print(f"\nBest parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        if defaults:
            print(f"\nStatic parameters (not tuned):")
            for key, value in defaults.items():
                print(f"  {key}: {value}")

        # Save best parameters
        best_params = {
            'algorithm': self.agent_type,
            'best_trial': study.best_trial.number,
            'best_metric': study.best_value,
            'best_params': study.best_params,
            'static_params': defaults or {},
            'best_agent': study.best_trial.user_attrs.get('agent_file'),
            'config': config,
            'timestamp': datetime.now().isoformat()
        }

        with open(self.best_params_file, 'w') as f:
            json.dump(best_params, f, indent=2)

        print(f"\nBest parameters saved to: {self.best_params_file}")
        print(f"All results saved to: {self.results_file}")

        # Generate report
        self._generate_report(study)

        return study

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function - must be implemented by subclass.
        """
        raise NotImplementedError("Subclass must implement _objective()")

    def _save_trial_result(self, trial, metric, final_value, sharpe, max_dd, agent_path):
        """Save individual trial result to JSON"""
        result = {
            'algorithm': self.agent_type,
            'trial_number': trial.number,
            'params': trial.params,
            'static_params': self.defaults or {},
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

    def _generate_report(self, study):
        """Generate CSV report"""
        trials_df = study.trials_dataframe()

        report_filename = f"{self.study_name}_report.csv"
        report_path = os.path.join(self.log_path, report_filename)
        trials_df.to_csv(report_path, index=False)

        print(f"Full report saved to: {report_path}")

        print("\nTop 5 trials:")
        top_trials = trials_df.nlargest(5, 'value')
        print(top_trials[['number', 'value']])
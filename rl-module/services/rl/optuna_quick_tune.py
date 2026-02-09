"""
Quick Optuna Tuning Configuration
Szybsze iteracje do początkowej eksploracji
"""

from optuna_ppo import OptunaPPOTuner

# Konfiguracja plików (jak w plgrid_train_ppo_simulation.py)
model_files = {
    "BTC": "btc_model_c_v1_1210_11.pkl"
}

train_data_files = {
    "BTC": "processed_ROBUST_set4_BTCUSDT-agent_1m_5m_2025-10-11_2d07d1b6-baf3-4ed8-bdc0-390156e92dc6.csv"
}

eval_data_files = {
    "BTC": "processed_ROBUST_set4_BTCUSDT-simulation_1m_2m_2025-10-11_8a076fe5-6d3c-4b1c-a490-9be3a3174517.csv"
}


class QuickTuner(OptunaPPOTuner):
    """
    Quick tuner z zawężoną przestrzenią poszukiwań
    NIE nadpisujemy objective() - tylko zmieniamy search space przez override suggest_* calls
    """

    def get_episode_timesteps(self, trial):
        """Zawężona lista dla quick exploration"""
        return trial.suggest_categorical('episode_timesteps', [512, 1024, 2048])

    def get_learning_rate(self, trial):
        """Tylko 3 wartości zamiast continuous"""
        return trial.suggest_categorical('learning_rate', [1e-4, 3e-4, 1e-3])

    def get_n_steps_multiplier(self, trial):
        """Tylko 2 opcje"""
        return trial.suggest_categorical('n_steps_multiplier', [2, 4])

    def get_batch_size(self, trial):
        """Tylko 2 opcje"""
        return trial.suggest_categorical('batch_size', [64, 128])

    def get_gamma(self, trial):
        """Tylko 2 wartości"""
        return trial.suggest_categorical('gamma', [0.97, 0.99])

    def get_ent_coef(self, trial):
        """Tylko 2 wartości"""
        return trial.suggest_categorical('ent_coef', [0.01, 0.05])

    def get_ml_threshold(self, trial):
        """3 wartości"""
        return trial.suggest_categorical('ml_threshold', [0.40, 0.45, 0.50])

    def get_total_timesteps(self, trial):
        """FIXED dla szybkości - nie sugerujemy, zawsze 500k"""
        return 500_000


if __name__ == "__main__":
    # Quick exploration - 10 trials, sequential
    quick_tuner = QuickTuner(
        model_files=model_files,
        train_data_files=train_data_files,
        eval_data_files=eval_data_files,
        study_name="quick_exploration_v3"
    )

    study = quick_tuner.run_study(
        n_trials=10,  # tylko 10 prób
        n_jobs=1,     # sequential (bezpieczne)
        timeout=None
    )

    print("\n" + "="*70)
    print("QUICK EXPLORATION DONE")
    print("="*70)
    print("\nNastępny krok:")
    print("1. Przejrzyj wyniki w optuna_studies/quick_exploration_v1_report.csv")
    print("2. Wybierz obiecujące zakresy parametrów")
    print("3. Uruchom pełny tuning z większą liczbą trials")
    print("\nPrzykład pełnego tuningu:")
    print("  python optuna_tune_ppo.py --n-trials 50 --n-jobs 4")
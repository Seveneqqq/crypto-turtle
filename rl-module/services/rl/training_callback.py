import sys
import json
import numpy as np
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback

class TrainingCallback(BaseCallback):
    def __init__(self, total_timesteps, num_cpu, base_dir, episode_timesteps, live=True):
        super().__init__()
        self.base_dir = Path(base_dir)
        self.episode_timesteps = episode_timesteps
        self.total_timesteps = total_timesteps
        self.num_envs = num_cpu
        self.live = live

        # Log paths
        self.log_dir = self.base_dir / "shared" / "agents" / "actual_training_logs"
        self.full_log_path = self.log_dir / "full_training.json"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Progress bar state
        self.progress_bar_length = 40
        self.last_progress = -1
        self.max_episodes = round((np.ceil(self.total_timesteps / self.episode_timesteps) + 15) // 16 * 16, 0)
        self.last_dump_epoch = -1

        if self.live:
            self.initialize_log_files()

    def _on_step(self) -> bool:
        """Obsługa paska postępu i wyzwalanie zapisu logów."""
        progress = int((self.num_timesteps / self.total_timesteps) * 10000)

        if progress != self.last_progress:
            self._print_progress_bar(progress)
            self.last_progress = progress

        if self.live and self.n_calls % (self.episode_timesteps + 1) == 0:
            self._dump_logs()

        return True

    def _print_progress_bar(self, progress: int):
        """Wyświetla pasek postępu w konsoli."""

        mean_reward = 0.0
        last_batch_avg = 0.0
        if self.model and getattr(self.model, "ep_info_buffer", None):
            rewards = [ep["r"] for ep in self.model.ep_info_buffer]
            if rewards:
                mean_reward = sum(rewards) / len(rewards)
                last_n = rewards[-self.num_envs:]
                if last_n:
                    last_batch_avg = sum(last_n) / len(last_n)

        pct = progress / 100
        filled = int(self.progress_bar_length * progress / 10000)
        bar = '#' * filled + '-' * (self.progress_bar_length - filled)

        sys.stdout.write(
            f"\rTraining PPO ({self.num_envs} envs): [{bar}] {pct}% | Mean Reward: {mean_reward:.4f} | Last Batch: {last_batch_avg:.4f}"
        )
        sys.stdout.flush()

    def _dump_logs(self):
        """Główna logika zapisu danych."""
        try:
            prev_epochs = self.training_env.get_attr("prev_epoch")
        except AttributeError:
            return

        valid_epochs = [p for p in prev_epochs if p and p.epoch > 0]
        if not valid_epochs:
            return

        for i, epoch_data in enumerate(valid_epochs):
            if hasattr(epoch_data, 'to_dict'):
                data = epoch_data.to_dict()
                data["process_id"] = i
                self._save_json(self.log_dir / f"proc_{i}.json", data)

        # Sprawdzenie czy zapisać globalny log (unikamy duplikatów)
        current_epoch_sum = sum(p.epoch for p in valid_epochs)
        if current_epoch_sum <= self.last_dump_epoch:
            return
        self.last_dump_epoch = current_epoch_sum

        # Agregacja i zapis full_training.json
        self._update_full_log(valid_epochs, current_epoch_sum)

    def _update_full_log(self, valid_epochs, epoch_sum):
        """Oblicza statystyki zbiorcze i aktualizuje główny plik historii."""
        num_procs = len(valid_epochs)

        def aggregate_stats(attr_suffix):
            values = [getattr(p, f"{stat}_{attr_suffix}") for p in valid_epochs for stat in
                      ['max', 'min', 'mean', 'end']]
            return {
                f"max_max_{attr_suffix}": max(getattr(p, f"max_{attr_suffix}") for p in valid_epochs),
                f"min_min_{attr_suffix}": min(getattr(p, f"min_{attr_suffix}") for p in valid_epochs),
                f"max_mean_{attr_suffix}": max(getattr(p, f"mean_{attr_suffix}") for p in valid_epochs),
                f"min_mean_{attr_suffix}": min(getattr(p, f"mean_{attr_suffix}") for p in valid_epochs),
                f"avg_end_{attr_suffix}": sum(getattr(p, f"end_{attr_suffix}") for p in valid_epochs) / num_procs,
                f"min_end_{attr_suffix}": min(getattr(p, f"end_{attr_suffix}") for p in valid_epochs),
                f"max_end_{attr_suffix}": max(getattr(p, f"end_{attr_suffix}") for p in valid_epochs),
            }

        new_entry = {
            "epoch_sum": epoch_sum,
            "avg_transactions": sum(p.buys + p.sells for p in valid_epochs) / num_procs,
            "avg_buys": sum(p.buys for p in valid_epochs) / num_procs,
            "avg_sells": sum(p.sells for p in valid_epochs) / num_procs,
            "portfolio": aggregate_stats('p'),
            "reward": aggregate_stats('r')
        }

        full_data = self._load_json(self.full_log_path, default={"config": {}, "history": []})

        # Aktualizacja config
        full_data["config"] = {
            "episode_length": self.episode_timesteps,
            "max_episodes": self.max_episodes,
            "num_processes": self.num_envs,
            "global_epochs": round(self.max_episodes / self.num_envs, 0),
        }

        if "history" not in full_data: full_data["history"] = []
        full_data["history"].append(new_entry)

        self._save_json(self.full_log_path, full_data)

    def initialize_log_files(self):
        """Inicjalizuje strukturę plików logów."""

        initial_data = {"config": {}, "history": []}
        self._save_json(self.full_log_path, initial_data)

        initial_proc_data = {
            "epoch": 0, "metrics": {"buys": 0, "sells": 0, "holds": 0},
            "portfolio": {"portfolio_values": [], "min_p": 0.0, "mean_p": 0.0, "max_p": 0.0},
            "reward": {"reward_values": [], "min_r": 0.0, "mean_r": 0.0, "max_r": 0.0}
        }

        for i in range(self.num_envs):
            proc_data = initial_proc_data.copy()
            proc_data["process_id"] = i
            self._save_json(self.log_dir / f"proc_{i}.json", proc_data)
        print(f"[Init] Utworzono pliki dla {self.num_envs} procesów")

    @staticmethod
    def _save_json(path, data):
        """Bezpieczny zapis JSON."""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"[Callback Error] Nie udało się zapisać {path.name}: {e}")

    @staticmethod
    def _load_json(path, default=None):
        """Bezpieczny odczyt JSON."""
        if not path.exists():
            return default if default is not None else {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[Callback Warning] Błąd odczytu {path.name}: {e}, tworzę nowy.")
            return default if default is not None else {}
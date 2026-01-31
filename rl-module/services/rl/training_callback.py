import sys
from stable_baselines3.common.callbacks import BaseCallback
import json
import os
import numpy as np

class TrainingCallback(BaseCallback):
    def __init__(self, total_timesteps, num_cpu, base_dir, episode_timesteps):
        super().__init__()
        self.base_dir = base_dir
        self.episode_timesteps = episode_timesteps
        self.total_timesteps = total_timesteps
        self.progress_bar_length = 40
        self.last_progress = -1
        self.num_envs = num_cpu
        self.threshold_history = {}
        self.FULL_LOG_PATH = os.path.join(self.base_dir,"shared","agents","actual_training_logs","full_training.json")

        self.max_episodes = round((np.ceil(self.total_timesteps / self.episode_timesteps) + 15) // 16 * 16,0)

        self.LOG_DIR = os.path.join(self.base_dir,"shared","agents","actual_training_logs")
        os.makedirs(self.LOG_DIR, exist_ok=True)
        self.initialize_log_files()

    def _on_step(self) -> bool:
        progress = int((self.num_timesteps / self.total_timesteps) * 10000)

        mean_reward = 0.0
        last_batch_avg = 0.0
        if self.model and hasattr(self.model, "ep_info_buffer") and self.model.ep_info_buffer:
            all_rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]

            if all_rewards:
                mean_reward = sum(all_rewards) / len(all_rewards)

            last_n_rewards = all_rewards[-self.num_envs:]

            if last_n_rewards:
                last_batch_avg = sum(last_n_rewards) / len(last_n_rewards)

        if progress != self.last_progress:
            bar_filled = int(self.progress_bar_length * progress / 10000)
            bar = '#' * bar_filled + '-' * (self.progress_bar_length - bar_filled)

            sys.stdout.write(
                f"\rTraining PPO ({self.num_envs} envs): [{bar}] {progress / 100}% | Mean Reward: {mean_reward:.4f} | Mean Reward Last Episodes: {last_batch_avg:.4f}"
            )
            sys.stdout.flush()
            self.last_progress = progress

        if self.n_calls % (self.episode_timesteps + 1) == 0:
            self._dump_logs()

        return True

    def _dump_logs(self):
        """Agregacja danych i zapis do plików JSON."""

        # 1. Pobranie danych z env (Poprawiona nazwa atrybutu)
        try:
            # Upewnij się, że w agent.py masz self.previous_epoch
            previous_epochs = self.training_env.get_attr("prev_epoch")
        except AttributeError:
            pass

        # --- 1. Zapis pojedynczych procesów (proc_X.json) ---
        valid_epochs = []

        for i, prev_epoch in enumerate(previous_epochs):
            if not prev_epoch or prev_epoch.epoch <= 0:
                continue

            valid_epochs.append(prev_epoch)

            # Zapis indywidualny
            if hasattr(prev_epoch, 'to_dict'):
                log_data = prev_epoch.to_dict()
                log_data["process_id"] = i

                filename = os.path.join(self.LOG_DIR, f"proc_{i}.json")
                try:
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(log_data, f)
                except Exception:
                    pass

        if not valid_epochs:
            return

        # Sprawdzenie czy to nowa epoka
        current_global_epoch_sum = sum(p.epoch for p in valid_epochs)
        # Opcjonalnie odkomentuj, jeśli nie chcesz dublować wpisów przy restarcie callbacka
        # if current_global_epoch_sum <= self.last_dump_epoch:
        #     return
        self.last_dump_epoch = current_global_epoch_sum

        # --- 2. Agregacja do full_training.json ---

        # Obliczenia statystyk
        num_procs = len(valid_epochs)

        avg_buys = sum(p.buys for p in valid_epochs) / num_procs
        avg_sells = sum(p.sells for p in valid_epochs) / num_procs
        avg_transactions = avg_buys + avg_sells

        # Portfolio stats
        max_max_p = max(p.max_p for p in valid_epochs)
        min_min_p = min(p.min_p for p in valid_epochs)
        max_mean_p = max(p.mean_p for p in valid_epochs)
        min_mean_p = min(p.mean_p for p in valid_epochs)

        avg_end_p = sum(p.end_p for p in valid_epochs) / num_procs
        min_end_p = min(p.end_p for p in valid_epochs)
        max_end_p = max(p.end_p for p in valid_epochs)

        # Reward stats
        max_max_r = max(p.max_r for p in valid_epochs)
        min_min_r = min(p.min_r for p in valid_epochs)
        max_mean_r = max(p.mean_r for p in valid_epochs)
        min_mean_r = min(p.mean_r for p in valid_epochs)

        avg_end_r = sum(p.end_r for p in valid_epochs) / num_procs
        min_end_r = min(p.end_r for p in valid_epochs)
        max_end_r = max(p.end_r for p in valid_epochs)

        new_entry = {
            "epoch_sum": current_global_epoch_sum,
            "avg_transactions": avg_transactions,
            "avg_buys": avg_buys,
            "avg_sells": avg_sells,

            "portfolio": {
                "max_max_p": max_max_p,
                "min_min_p": min_min_p,
                "max_mean_p": max_mean_p,
                "min_mean_p": min_mean_p,
                "avg_end_p": avg_end_p,
                "min_end_p": min_end_p,
                "max_end_p": max_end_p
            },

            "reward": {
                "max_max_r": max_max_r,
                "min_min_r": min_min_r,
                "max_mean_r": max_mean_r,
                "min_mean_r": min_mean_r,
                "avg_end_r": avg_end_r,
                "min_end_r": min_end_r,
                "max_end_r": max_end_r
            }
        }

        # --- BEZPIECZNY ODCZYT I AKTUALIZACJA PLIKU ---
        full_data = {"config": {}, "history": []}

        if os.path.exists(self.FULL_LOG_PATH):
            try:
                with open(self.FULL_LOG_PATH, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    # Upewniamy się, że to słownik
                    if isinstance(content, dict):
                        full_data = content
            except Exception as e:
                print(f"[Callback Warning] Nie udało się odczytać full_training.json (tworzę nowy): {e}")

        # Zabezpieczenie przed brakiem kluczy (TO NAPRAWIA BŁĄD)
        if "history" not in full_data:
            full_data["history"] = []
        if "config" not in full_data:
            full_data["config"] = {}

        # Aktualizacja configu
        full_data["config"] = {
            "episode_length": self.episode_timesteps,
            "max_episodes": self.max_episodes,
            "num_processes": self.num_envs,
            "global_epochs": round(self.max_episodes / self.num_envs,0),
        }

        # Dodanie nowego wpisu
        full_data["history"].append(new_entry)

        try:
            with open(self.FULL_LOG_PATH, 'w', encoding='utf-8') as f:
                json.dump(full_data, f, indent=4)
        except Exception as e:
            print(f"[Callback Error] Błąd zapisu full_training.json: {e}")

    def initialize_log_files(self):
        """
        Tworzy pliki startowe: proc_0.json... oraz full_training.json
        z nową strukturą danych.
        """
        # 1. Upewnij się, że folder istnieje
        os.makedirs(self.LOG_DIR, exist_ok=True)
        print(f"[Init] Inicjalizacja {self.num_envs} plików logów w: {self.LOG_DIR}")

        # --- 1. Inicjalizacja full_training.json (Log Globalny) ---
        # Tworzymy go tylko jeśli nie istnieje, żeby nie nadpisać historii przy restarcie skryptu
        if os.path.exists(self.FULL_LOG_PATH):
            initial_full_data = {
                "config": {
                    "episode_length": self.episode_timesteps,
                    "max_episodes": self.max_episodes,
                    "num_processes": self.num_envs,
                    "global_epochs": round(self.max_episodes / self.num_envs,0),
                },
                "history": []
            }
            try:
                with open(self.FULL_LOG_PATH, 'w', encoding='utf-8') as f:
                    json.dump(initial_full_data, f, indent=4)
                print(f"[Init] Utworzono nowy plik globalny: {self.FULL_LOG_PATH}")
            except Exception as e:
                print(f"[Init Error] Nie udało się utworzyć full_training.json: {e}")

        # --- 2. Inicjalizacja proc_X.json (Logi Procesów) ---
        # Te pliki są nadpisywane na starcie, by Dashboard widział, że procesy są "czyste"
        for i in range(self.num_envs):
            file_path = os.path.join(self.LOG_DIR, f"proc_{i}.json")

            # Nowa struktura zgodna z PreviousEpoch.to_dict()
            initial_data = {
                "process_id": i,
                "epoch": 0,  # Stan zerowy
                "metrics": {
                    "buys": 0,
                    "sells": 0,
                    "holds": 0
                },
                "portfolio": {
                    "portfolio_values": [],
                    "min_p": 0.0,
                    "mean_p": 0.0,
                    "max_p": 0.0,
                    "end_p": 0.0
                },
                "reward": {
                    "reward_values": [],
                    "min_r": 0.0,
                    "mean_r": 0.0,
                    "max_r": 0.0,
                    "end_r": 0.0
                }
            }

            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(initial_data, f, indent=4)
            except Exception as e:
                print(f"[Init Error] Błąd inicjalizacji {file_path}: {e}")

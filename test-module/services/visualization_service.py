import os
import json
import gzip
import pandas as pd
from .test.charts import ChartGenerator


class VisualizationService:
    def __init__(self):
        self.chart_generator = ChartGenerator()
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.gzip_path = os.path.join(self.base_dir, "shared", "simulation","outputs")
    def visualize(self, simulation_files, sim_path):
        """
        Główna metoda koordynująca wizualizację.
        :param simulation_files: Nazwa pliku .json.gz z danymi symulacji.
        :param sim_path: Ścieżka do katalogu z wynikami symulacji.
        """
        filepath = os.path.join(self.gzip_path, simulation_files)

        # 1. Wczytanie głównego pliku danych (JSON GZIP)
        try:
            with gzip.open(filepath, "rt", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[Visualization] Critical Error - Cannot load JSON data: {e}")
            return

        # 2. Wykrycie trybu (Agent RL vs Model ML)
        # Sprawdzamy, która sekcja jest dostępna w danych
        if "agent" in data:
            mode_key = "agent"
            logs_filename = "agent_transaction_logs.txt"
        elif "model" in data:
            mode_key = "model"
            logs_filename = "model_transaction_logs.txt"
        else:
            print("[Visualization] Error: Unknown simulation mode in data file.")
            return

        sim_data = data[mode_key]

        # 3. Ekstrakcja danych wektorowych
        portfolio_values = sim_data.get("portfolio_values", [])
        metrics = sim_data.get("metrics", {})

        # Sekcja HODL (Ceny referencyjne)
        crypto_ref_values = data.get("hodl", {}).get("crypto_ref_values", {})

        # 4. Obsługa osi czasu (Timestamps)
        # UWAGA: Wykrywanie błędu w simulation_service (zapis długości zamiast listy)
        timestamps_raw = sim_data.get("timestamps", 0)

        if isinstance(timestamps_raw, int):
            # Jeśli mamy tylko liczbę kroków -> generujemy indeksy [0, 1, 2...]
            # (Wykres będzie pokazywał numer kroku zamiast daty)
            timestamps = list(range(timestamps_raw))
        elif isinstance(timestamps_raw, list):
            # Jeśli mamy listę -> próbujemy konwertować na datetime dla ładnych wykresów
            try:
                timestamps = pd.to_datetime(timestamps_raw)
            except Exception:
                timestamps = timestamps_raw
        else:
            timestamps = []

        # 5. Wczytanie logów transakcji do DataFrame (Szybkie!)
        logs_path = os.path.join(sim_path, logs_filename)
        df_logs = pd.DataFrame()

        if os.path.exists(logs_path):
            try:
                # read_json z lines=True idealnie pasuje do Twojego formatu zapisu logów
                df_logs = pd.read_json(logs_path, lines=True)
            except ValueError:
                # Obsługa pustego pliku
                pass
            except Exception as e:
                print(f"[Visualization] Warning: Could not read transaction logs: {e}")

        # 6. Generowanie wykresu
        # Usuwamy rozszerzenia .json.gz, aby uzyskać czystą nazwę pliku
        base_filename = simulation_files.replace(".json.gz", "").replace(".json", "")

        try:
            output_chart_path = self.chart_generator.generate_chart(
                timestamps=timestamps,
                portfolio_values=portfolio_values,
                crypto_ref_values=crypto_ref_values,
                logs=df_logs,  # Przekazujemy DataFrame!
                metrics=metrics,
                filename=base_filename,
                output_dir=sim_path
            )
            print(f"[Visualization] Success! Chart saved to: {output_chart_path}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Visualization] Error during chart generation: {e}")
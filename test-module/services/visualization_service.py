import os
import gzip
import json
from datetime import datetime, timedelta
from services.test.charts import Charts
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt

class VisualizationService:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.SIM_FILE_DIR = os.path.join(base_dir, "shared", "simulation", "outputs")
        self.VIS_DIR = os.path.join(base_dir, "shared", "simulation", "visualizations")
        os.makedirs(self.VIS_DIR, exist_ok=True)
        self.simulations_data = None
        self.files_to_load = None
        self.charts = Charts()

    def visualize(self,
                  rabbit_msg_id=None,
                  simulation_files=None,
                  sim_path=None):
        """Wczytuje dane z plików gzip tylko z SIM_FILE_DIR."""
        self.files_to_load = self._collect_gzip_files(simulation_files)

        self.simulations_data = []

        for path in self.files_to_load:
            try:
                sim_data = self._load_gzip_file(path)
                self.simulations_data.append(sim_data)
                print(f"✅ Wczytano {path}")
            except Exception as e:
                print(f"⚠️ Błąd przy wczytywaniu {path}: {e}")
        print(f"📦 Wczytano {len(self.simulations_data)} symulacji.")


        if len(self.simulations_data)==1 and sim_path is not None:
            sim_data = self.simulations_data[0]
            self.sim_visualization(sim_data,sim_path)
        if sim_path is None:
            vis_path = self.build_filename()
            self.vis_loop(self.simulations_data,vis_path)


    def sim_visualization(self,data,sim_path):
        model_plot_path = os.path.join(sim_path, "model_plot.png")
        agent_plot_path = os.path.join(sim_path, "agent_plot.png")
        sim_plot_path = os.path.join(sim_path, "sim_plot.png")
        if data["config"]["sim_mode"] in ("ml", "mlrl"):
            self.charts.draw_chart(data["model"]["portfolio_values"],
                              data["hodl"]["crypto_ref_values"],
                              data["model"]["timestamps"],
                              data["model"]["metrics"],
                              data["config"]["symbols"],
                              model_plot_path)

        if data["config"]["sim_mode"] in ("rl", "mlrl"):
            self.charts.draw_chart(data["agent"]["portfolio_values"],
                              data["hodl"]["crypto_ref_values"],
                              data["agent"]["timestamps"],
                              data["agent"]["metrics"],
                              data["config"]["symbols"],
                              agent_plot_path)
        if data["config"]["sim_mode"] == "mlrl":
            self.charts.draw_common_chart(data["model"]["portfolio_values"],
                                     data["agent"]["portfolio_values"],
                                     data["model"]["timestamps"],
                                     data["agent"]["timestamps"],
                                     data["model"]["metrics"],
                                     data["agent"]["metrics"],
                                     sim_plot_path)

    def vis_loop(self, data, output_pdf):


        with PdfPages(output_pdf) as pdf:
            for i, item in enumerate(data):
                try:
                    file_name = os.path.basename(self.files_to_load[i])
                    fig = self.charts.draw_universal_chart(item, pdf_title=f"Symulacja [{i + 1}/{len(data)}], plik: {file_name}")

                    pdf.savefig(fig)
                    plt.close(fig)

                    # fig = self.charts.draw_crypto_percentage_chart(item,False)
                    # # dodaj stronę do PDF
                    # pdf.savefig(fig)
                    # plt.close(fig)

                    print(f"[{i + 1}/{len(data)}] Dodano stronę dla: {file_name}")
                except Exception as e:
                    print(f"Błąd przy wizualizacji {i + 1}: {e}")

        print(f"✅ Zapisano wszystkie wizualizacje do pliku pdf")

    def build_filename(self):
        date_str = datetime.now().strftime("%d_%m")
        existing_files = [
            f for f in os.listdir(self.VIS_DIR)
            if f.startswith("vis_") and f"_{date_str}_" in f and f.endswith(".pdf")
        ]

        versions = []
        for f in existing_files:
            try:
                base = os.path.splitext(os.path.splitext(f)[0])[0]
                v_part = base.split("_v")[-1].split("_")[0]
                versions.append(int(v_part))
            except ValueError:
                continue
        next_version = max(versions, default=0) + 1
        v_p = f"vis_{date_str}_v{next_version}.pdf"
        vis_path = os.path.join(self.VIS_DIR, v_p)

        return vis_path

    def _load_gzip_file(self, path: str):
        """Wczytuje jeden plik gzip i zwraca dane JSON."""
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)

    def _collect_gzip_files(self, simulation_files):
        """Zbiera listę plików gzip tylko z folderu self.SIM_FILE_DIR."""
        if simulation_files is None:
            return []

        if isinstance(simulation_files, str):
            full_path = os.path.join(self.SIM_FILE_DIR, simulation_files)

            if os.path.isdir(full_path):
                # folder → wszystkie .gz w środku
                return sorted([
                    os.path.join(full_path, f)
                    for f in os.listdir(full_path)
                    if f.endswith(".json.gz")
                ])

            elif full_path.endswith(".json.gz") and os.path.exists(full_path):
                return [full_path]

            else:
                raise ValueError(f"Nieprawidłowa ścieżka lub plik nie istnieje: {simulation_files}")

        elif isinstance(simulation_files, list):
            paths = []
            for f in simulation_files:
                full_path = os.path.join(self.SIM_FILE_DIR, f)
                if full_path.endswith(".json.gz") and os.path.exists(full_path):
                    paths.append(full_path)
                else:
                    print(f"⚠️ Ignorowany niepoprawny plik: {f}")
            return paths

        else:
            raise TypeError("simulation_files musi być stringiem, listą lub folderem w SIM_FILE_DIR.")

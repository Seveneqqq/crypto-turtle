import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

class Charts:
    def __init__(self):
        self.portfolio_values = {}

    @staticmethod
    def draw_universal_chart(data, pdf_title):
        """
        Tworzy uniwersalny wykres dla trybów ML, RL lub MLRL,
        bez błędów gdy brakuje agent/model/crypto_ref_values
        """
        # Bezpieczne pobranie metryk
        agent_metrics = data.get("agent", {}).get("metrics", {})
        model_metrics = data.get("model", {}).get("metrics", {})

        agent_rate = agent_metrics.get("score", 0)
        model_rate = model_metrics.get("score", 0)

        sim_mode = data.get("config", {}).get("sim_mode", "")
        symbols = data.get("config", {}).get("symbols", [])
        crypto_ref_values = data.get("hodl", {}).get("crypto_ref_values", {})

        fig, (ax_title, ax) = plt.subplots(
            2, 1,
            gridspec_kw={'height_ratios': [0.05, 0.95]},  # 10% na tytuł, 90% na wykres
            figsize=(12, 7)
        )
        ax_title.axis("off")  # wyłącz osie
        ax_title.text(0.5, 0.5, pdf_title, ha='center', va='center', fontsize=12, fontweight='bold')

        # --- Wartości portfeli ---
        if sim_mode in ("ml", "mlrl") and "model" in data:
            model_portfolio = data["model"].get("portfolio_values", [])
            if model_portfolio:
                ax.plot(range(len(model_portfolio)), model_portfolio, label="Model (ML)", color="blue")

        if sim_mode in ("rl", "mlrl") and "agent" in data:
            agent_portfolio = data["agent"].get("portfolio_values", [])
            if agent_portfolio:
                ax.plot(range(len(agent_portfolio)), agent_portfolio, label="Agent (RL)", color="red")

        # --- HODL (tylko jeśli jedna kryptowaluta i istnieją dane) ---
        if len(symbols) == 1 and symbols[0] in crypto_ref_values:
            hodl_values = crypto_ref_values.get(symbols[0], [])
            if hodl_values:
                ax.plot(range(len(hodl_values)), hodl_values, linestyle="--", color="gray", label=f"HODL {symbols[0]}")

        # --- Ustawienia wykresu ---
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Wartość portfela (USD)")

        title_map = {
            "ml": f"Model ML [{model_rate}] – Symulacja wartości portfela",
            "rl": f"Agent RL [{agent_rate}] – Symulacja wartości portfela",
            "mlrl": f"Porównanie: Model ML [{model_rate}] vs Agent RL [{agent_rate}]"
        }
        ax.set_title(title_map.get(sim_mode, "Symulacja portfela"))

        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        return fig

    @staticmethod
    def draw_crypto_percentage_chart(data, rlagent: bool = False):
        """
        Rysuje wykres wartości portfela, gotówki (USD) i poszczególnych kryptowalut.
        Zakłada strukturę:
          data["config"]["symbols"] -> ["BTC", "ETH", ...]
          data["model" lub "agent"]["holdings"] -> [[btc_t0, eth_t0, ...], [btc_t1, eth_t1, ...], ...]
        """
        key = "agent" if rlagent else "model"
        section = data[key]

        portfolio_values = section["portfolio_values"]
        holdings = section["holdings"]
        symbols = data["config"]["symbols"]

        # transpozycja holdings: [ [btc_t0, eth_t0], [btc_t1, eth_t1] ] -> {BTC: [..], ETH: [..]}
        holdings_by_symbol = {sym: [row[i] for row in holdings] for i, sym in enumerate(symbols)}

        # Oblicz USD (gotówka)
        usd_values = []
        for i in range(len(portfolio_values)):
            total_crypto = sum(holdings[i])  # suma wartości kryptowalut w danym kroku
            usd = portfolio_values[i] - total_crypto
            usd_values.append(usd)

        # Rysowanie wykresu
        fig, ax = plt.subplots(figsize=(12, 7))
        x = range(len(portfolio_values))

        # linie
        ax.plot(x, portfolio_values, color="black", linewidth=2, label="Portfolio total")
        ax.plot(x, usd_values, color="gray", linestyle="--", label="USD (cash)")

        for sym, vals in holdings_by_symbol.items():
            ax.plot(x, vals, label=sym)

        # opis i legenda
        mode_name = "Agent RL" if rlagent else "Model ML"
        ax.set_title(f"Skład portfela — {mode_name}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Kroki / Timestamp")
        ax.set_ylabel("Wartość (USD)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        return fig

    # metody dla simulation_service (na razie zostają)

    @staticmethod
    def draw_common_chart(model_portfolio_values, agent_portfolio_values, model_timestamps, agent_timestamps, model_metrics, agent_metrics, plot_path):
        fig, (ax, ax_text) = plt.subplots(
            2, 1,
            gridspec_kw={'height_ratios': [5, 1]},
            figsize=(12, 9)
        )

        # --- Wykres wartości portfela ---
        ax.plot(range(0,model_timestamps), model_portfolio_values, label="Model DecisionEngine", color="blue")
        ax.plot(range(0,agent_timestamps), agent_portfolio_values, label="Agent RL", color="red")

        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Wartość portfela (USD)")
        ax.set_title("Porównanie wartości portfela: Model vs Agent RL")
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.legend()

        # --- Tekst porównawczy bez LaTeX ---
        ax_text.axis("off")

        # Pozycje pionowe
        y_start = 0.9
        delta_y = 0.12

        metrics_list = ["total_return", "max_portfolio_growth", "max_portfolio_drop",
                        "transaction_count", "avg_transactions_per_day",
                        "cagr", "max_drawdown", "sharpe_ratio", "sortino_ratio", "calmar_ratio", "score"]

        metric_names = ["Łączny zwrot", "Maksymalny wzrost portfela", "Maksymalny spadek portfela",
                        "Liczba transakcji", "Śr. transakcji/dzień",
                        "CAGR", "Max Drawdown", "Sharpe ratio", "Sortino ratio", "Calmar ratio", "Score"]

        for i, (key, name) in enumerate(zip(metrics_list, metric_names)):
            m_val = model_metrics.get(key, 0)
            a_val = agent_metrics.get(key, 0)

            if m_val > a_val:
                winner = "Model"
                diff = (m_val - a_val) / a_val * 100 if a_val != 0 else 0
            elif a_val > m_val:
                winner = "Agent RL"
                diff = (a_val - m_val) / m_val * 100 if m_val != 0 else 0
            else:
                winner = "Remis"
                diff = 0

            # Opis
            ax_text.text(0.2, y_start - i * delta_y, f"{name}:", ha="right", va="center", fontsize=10)
            # Model (niebieski)
            ax_text.text(0.25, y_start - i * delta_y, f"{m_val:.2f}", color="blue", ha="center", va="center",
                         fontsize=10)
            # Agent (czerwony)
            ax_text.text(0.4, y_start - i * delta_y, f"{a_val:.2f}", color="red", ha="center", va="center",
                         fontsize=10)
            # Zwycięzca
            if winner == "Model":
                ax_text.text(0.55, y_start - i * delta_y, f"{winner} (+{diff:.2f}%)", color="blue", ha="left",
                             va="center",
                             fontsize=10)
            else:
                ax_text.text(0.55, y_start - i * delta_y, f"{winner} (+{diff:.2f}%)", color="red", ha="left",
                             va="center",
                             fontsize=10)
        plt.tight_layout()
        plt.show()
        plt.savefig(plot_path, format="png")
        plt.close()


    @staticmethod
    def draw_chart(portfolio_values, crypto_ref_values, timestamps, metrics, symbols, plot_path):

        fig, (ax, ax_text) = plt.subplots(
            2, 1,
            gridspec_kw={'height_ratios': [5, 1]},  # proporcja: 5 - wykres, 1 - tekst
            figsize=(12, 9)
        )
        print(f"timestamps: {timestamps}")
        print(f"portfolio_values: {len(portfolio_values)}")
        # --- Wykres ---
        ax.plot(range(0, timestamps), portfolio_values, label="Wartość portfela (USD)")

        if len(symbols) == 1:
            for symbol in symbols:
                ax.plot(range(0, timestamps), crypto_ref_values[symbol], linestyle='--',
                        label=f"HODL {symbol}")
        title = "Model ML"
        if plot_path == "agent_plot.png": title = "Agent RL"

        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Wartość portfela (USD)")
        ax.set_title(f"[{title}] Symulacja wartości portfela (Wynik: {metrics['score']} pkt)")
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.legend()

        # --- Podział metryk na 3 bloki ---
        block_left = [
            # f"Pliki z danymi: {data_files}",
            # f"Pliki z modelami: {models}",
            # f"Pliki z agentami: {agents}",
            f"Tu będą pliki jak wymyślę jak to zawijać"
        ]

        block_center = [
            f"Łączny zwrot: {metrics['total_return']:.2f}%",
            f"Maksymalny wzrost portfela: {metrics['max_portfolio_growth']:.2f}%",
            f"Maksymalny spadek portfela: {metrics['max_portfolio_drop']:.2f}%",
            f"Ilość transakcji: {metrics['transaction_count']}",
            f"Śr. transakcji/dzień: {metrics['avg_transactions_per_day']:.2f}"
        ]

        block_right = [
            f"CAGR: {metrics['cagr']:.2f}%",
            f"Max Drawdown: {metrics['max_drawdown']:.2f}%",
            f"Calmar ratio: {metrics['calmar_ratio']:.2f}",
            f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}",
            f"Sortino ratio: {metrics['sortino_ratio']:.2f}"
        ]

        # --- Rysowanie bloków pod wykresem ---
        ax_text.axis("off")

        # Lewy blok (x=0.2)
        ax_text.text(
            0.2, 0.5, "\n".join(block_left),
            ha="center", va="center", fontsize=10,
            bbox=dict(facecolor="whitesmoke", edgecolor="gray", boxstyle="round,pad=0.5")
        )

        # Środkowy blok pusty (x=0.5)
        ax_text.text(
            0.5, 0.5, "\n".join(block_center),
            ha="center", va="center", fontsize=10,
            bbox=dict(facecolor="whitesmoke", edgecolor="gray", boxstyle="round,pad=0.5")
        )

        # Prawy blok (x=0.8)
        ax_text.text(
            0.8, 0.5, "\n".join(block_right),
            ha="center", va="center", fontsize=10,
            bbox=dict(facecolor="whitesmoke", edgecolor="gray", boxstyle="round,pad=0.5")
        )

        plt.tight_layout()
        plt.show()
        plt.savefig(plot_path, format="png")
        plt.close()
        return metrics
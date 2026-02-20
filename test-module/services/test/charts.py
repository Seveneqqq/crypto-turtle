import matplotlib

# Ustawienie backendu 'Agg' jest krytyczne dla Dockera (brak GUI)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np


class ChartGenerator:
    def generate_chart(self, timestamps, portfolio_values, crypto_ref_values, logs, metrics, filename, output_dir):
        """
        Generuje dashboard symulacji.
        :param logs: Może być listą słowników lub pd.DataFrame
        """

        # 1. PRZYGOTOWANIE DANYCH
        # Zabezpieczenie: jeśli logs to lista, konwertujemy na DataFrame
        if isinstance(logs, list):
            df_logs = pd.DataFrame(logs)
        else:
            df_logs = logs.copy()

        # Konwersja timestampów logów jeśli trzeba (zakładamy, że są zgodne z osią X)
        # Jeśli timestamps to obiekty datetime, upewnij się, że w df_logs też są datetime.

        # 2. USTAWIENIA STYLU
        plt.style.use('dark_background')
        # Kolory dla poszczególnych kryptowalut (cykl kolorów)
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        # 3. UKŁAD WYKRESU (3 wiersze: Portfel, Ceny, Metryki)
        # height_ratios: Metryki na dole zajmują mniej miejsca
        fig, (ax_port, ax_metrics) = plt.subplots(
            2, 1,
            figsize=(16, 10),
            sharex=False,  # Metrics nie ma osi czasu
            gridspec_kw={'height_ratios': [3, 1]}
        )
        # fig, (ax_port, ax_price, ax_metrics) = plt.subplots(
        #     3, 1,
        #     figsize=(16, 14),
        #     sharex=False,  # Metrics nie ma osi czasu
        #     gridspec_kw={'height_ratios': [3,2, 1]}
        # )

        # Aby Portfel i Ceny miały wspólną oś X (zoom działa na obu)
        #ax_port.get_shared_x_axes().joined(ax_port, ax_price)
        #ax_price.set_xticklabels([])  # Ukryj etykiety X dla środkowego wykresu

        # --- WYKRES 1: PORTFOLIO VALUE ---
        self._plot_portfolio(ax_port, timestamps, portfolio_values,crypto_ref_values)

        # --- WYKRES 2: CENY I TRANSAKCJE ---
        #self._plot_assets_and_signals(ax_price, timestamps, crypto_ref_values, df_logs, colors)

        # --- SEKCJA 3: METRYKI ---
        self._render_metrics(ax_metrics, metrics)

        # 4. ZAPIS I CZYSZCZENIE
        plt.tight_layout()
        output_path = f"{output_dir}/{filename}.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)  # Zwolnienie pamięci RAM

        return output_path

    def _plot_portfolio(self, ax, timestamps, values, crypto_ref_values):
        """Rysuje wykres wartości portfela vs HODL."""

        # 1. Główna linia (Twoja Strategia)
        # Pogrubiłem ją (linewidth=2) żeby się wyróżniała
        ax.plot(timestamps, values, color='#00ffcc', linewidth=1, label='Strategy (Total)')
        ax.fill_between(timestamps, values, alpha=0.1, color='#00ffcc')

        # 2. Linie referencyjne (HODL) dla każdego symbolu
        # Definiujemy kolory dla linii HODL (np. pomarańczowy dla BTC, fiolet dla ETH...)
        colors = ['#ff9900', '#ff3333', '#ffff00', '#cc00ff', '#99ff33']

        for i, (symbol, ref_vals) in enumerate(crypto_ref_values.items()):
            # Wybór koloru cyklicznie
            color = colors[i % len(colors)]

            # Zabezpieczenie długości tablic (na wypadek różnic o 1 tick)
            limit = min(len(timestamps), len(ref_vals))

            ax.plot(
                timestamps[:limit],
                ref_vals[:limit],
                color=color,
                linestyle='--',  # Linia przerywana odróżnia benchmark od strategii
                linewidth=1,
                alpha=0.6,  # Lekko przezroczyste, żeby nie zagłuszały wykresu
                label=f'HODL {symbol}'
            )
        all_values = values + [val for sublist in crypto_ref_values.values() for val in sublist]
        if all_values:
            y_min = min(all_values)
            y_max = max(all_values)
            # Dodajemy 5% marginesu góra/dół, żeby wykres nie dotykał krawędzi
            margin = (y_max - y_min) * 0.05
            ax.set_ylim(y_min - margin, y_max + margin)
        # 3. Stylizacja i Legenda
        ax.set_title("Portfolio Performance vs HODL", fontsize=14, fontweight='bold', color='white')
        ax.set_ylabel("Value (USDT)", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14, colors='#dddddd')

        ax.grid(True, linestyle='--', alpha=0.2)

        # Legenda automatycznie doda nowe wpisy
        ax.legend(loc='upper left', frameon=True, facecolor='#1a1a1a', edgecolor='none', fontsize=12)

    def _plot_assets_and_signals(self, ax, timestamps, crypto_refs, df_logs, colors):
        """Rysuje ceny aktywów oraz nakłada punkty kupna/sprzedaży."""

        symbols = list(crypto_refs.keys())

        for i, symbol in enumerate(symbols):
            # Dobieramy kolor dla danego symbolu
            color = colors[i % len(colors)]
            ref_values = crypto_refs[symbol]

            # Dopasowanie długości (na wypadek różnic o 1 tick)
            limit = min(len(timestamps), len(ref_values))

            # Rysowanie linii ceny (Referencyjna wartość HODL lub Cena)
            ax.plot(timestamps[:limit], ref_values[:limit], label=f"{symbol}", color=color, alpha=0.7, linewidth=1)

            # --- RYSOWANIE TRANSAKCJI (WEKTOROWO) ---
            if not df_logs.empty:
                # Filtrujemy DataFrame dla konkretnego symbolu
                mask_symbol = df_logs['symbol'] == symbol

                # KUPNO (Zielony trójkąt w górę)
                buys = df_logs[mask_symbol & (df_logs['action'] == 'buy')]
                if not buys.empty:
                    self._scatter_transactions(ax, timestamps, buys, marker='^', color='#00ff00', label=None)

                # SPRZEDAŻ (Czerwony trójkąt w dół)
                sells = df_logs[mask_symbol & (df_logs['action'] == 'sell')]
                if not sells.empty:
                    self._scatter_transactions(ax, timestamps, sells, marker='v', color='#ff0066', label=None)

        ax.set_title("💰 Asset Prices & Transactions", fontsize=14, fontweight='bold')
        ax.set_ylabel("Asset Value (Ref)", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.15)
        ax.legend(loc='upper left', frameon=True, facecolor='#1a1a1a', edgecolor='none', fontsize=8)

        # Formatowanie daty na osi X (jeśli timestamps to datetime)
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        # ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    def _scatter_transactions(self, ax, timestamps, log_subset, marker, color, label):
        """Pomocnicza funkcja do szybkiego rysowania punktów."""
        # Musimy zmapować timestamp z logów na oś X wykresu.
        # Wersja A: Jeśli log['timestamp'] to index (int)
        try:
            indices = log_subset['timestamp'].values.astype(int)
            # Filtrujemy indeksy, które wychodzą poza zakres
            valid_mask = indices < len(timestamps)
            indices = indices[valid_mask]
            prices = log_subset['price'].values[valid_mask]  # Zakładając, że loguje cenę aktywa a nie wartość portfela

            # Pobieramy daty odpowiadające indeksom
            dates = [timestamps[i] for i in indices]

            # Rysujemy
            ax.scatter(dates, prices, marker=marker, color=color, s=60, edgecolors='black', linewidth=0.5, zorder=5,
                       alpha=0.9, label=label)

        except Exception as e:
            print(f"[Chart Error] Could not plot transactions: {e}")

    def _render_metrics(self, ax, metrics):
        """Renderuje tabelkę z metrykami zamiast wykresu."""
        ax.axis('off')  # Wyłączamy osie

        # Tytuł sekcji
        ax.text(0.01, 0.9, "SIMULATION METRICS", transform=ax.transAxes,
                fontsize=16, fontweight='bold', color='#aaaaaa')

        # Dzielimy metryki na dwie kolumny
        items = list(metrics.items())
        mid = (len(items) + 1) // 2
        col1 = items[:mid]
        col2 = items[mid:]

        # Funkcja pomocnicza do budowania stringa
        def build_col_text(items_list):
            text = ""
            for k, v in items_list:
                # Skracanie nazw kluczy i formatowanie liczb
                clean_key = k.replace("_", " ").title()[:20]
                if isinstance(v, float):
                    val_str = f"{v:.4f}"
                else:
                    val_str = str(v)
                text += f"{clean_key:<22} : {val_str}\n"
            return text

        # Rysowanie tekstu w dwóch kolumnach (monospace font jest kluczowy do wyrównania)
        ax.text(0.02, 0.75, build_col_text(col1), transform=ax.transAxes,
                fontsize=14, family='monospace', color='white', verticalalignment='top')

        ax.text(0.52, 0.75, build_col_text(col2), transform=ax.transAxes,
                fontsize=14, family='monospace', color='white', verticalalignment='top')
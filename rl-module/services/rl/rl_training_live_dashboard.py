import glob
import dash
import time
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import json
import os
import pandas as pd
import numpy as np

# --- KONFIGURACJA ---
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(base_dir)
LOG_DIR = os.path.join(base_dir, "shared", "agents", "actual_training_logs")
os.makedirs(LOG_DIR, exist_ok=True)
UPDATE_INTERVAL_MS = 4000
TARGET_EPOCHS = 100

app = dash.Dash(__name__)

# --- STYLE ---
COLORS = {
    'bg': '#111111', 'card': '#1e1e1e', 'text': '#e0e0e0', 'accent': '#00d4ff',
    'success': '#00e676', 'warning': '#ff9100', 'danger': '#ff1744'
}

btn_style = {
    'backgroundColor': '#333333', 'color': 'white', 'border': '1px solid #555',
    'padding': '10px 20px', 'margin': '10px 5px 25px', 'borderRadius': '5px',
    'cursor': 'pointer', 'fontSize': '12px', 'fontFamily': 'Roboto, sans-serif'
}

btn_active_style = {
    **btn_style,
    'backgroundColor': '#00d4ff', 'color': '#000000', 'fontWeight': 'bold',
    'border': '1px solid #00d4ff', 'boxShadow': '0 0 10px #00d4ff'
}

card_style = {
    'backgroundColor': COLORS['card'], 'borderRadius': '10px', 'padding': '10px',
    'margin': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.3)', 'width': '30%'
}


# --- FUNKCJE DANYCH ---
def get_full_training_data():
    """Pobiera zagregowane dane globalne."""
    path = os.path.join(LOG_DIR, "full_training.json")
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return None


def get_process_data():
    """Pobiera dane z poszczególnych procesów (Grid)."""
    files = glob.glob(os.path.join(LOG_DIR, "proc_*.json"))
    try:
        files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    except:
        files.sort()

    data = []
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                data.append(json.load(fp))
        except:
            pass
    return data


# --- NOWA SEKCJA GLOBALNA (LAYOUT ZACHOWANY) ---
global_section = html.Div([
    html.Div([
        # LEWY PANEL
        html.Div([
            html.Div([
                html.Button("📈 Reward Sum", id='btn-g1-rew', n_clicks=0, style=btn_active_style),
                html.Button("💰 Port. Sum", id='btn-g1-port', n_clicks=0, style=btn_style),
                html.Button("🌊 Port. Vol", id='btn-g1-heat', n_clicks=0, style=btn_style),
                html.Button("📊 Trans. Hist", id='btn-g1-hist', n_clicks=0, style=btn_style),
                html.Button("📉 Rew. Vol", id='btn-g1-none', n_clicks=0, style=btn_style),
            ], style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '10px'}),
            dcc.Graph(id='global-graph-1', style={'height': '400px'}, config={'displayModeBar': False}),
            dcc.Store(id='store-g1', data='reward')
        ], style={**card_style, 'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        # PRAWY PANEL
        html.Div([
            html.Div([
                html.Button("💰 Port. Sum", id='btn-g2-port', n_clicks=0, style=btn_active_style),
                html.Button("📈 Reward Sum", id='btn-g2-rew', n_clicks=0, style=btn_style),
                html.Button("🌊 Port. Vol", id='btn-g2-heat', n_clicks=0, style=btn_style),
                html.Button("📊 Trans. Hist", id='btn-g2-hist', n_clicks=0, style=btn_style),
                html.Button("📉 Rew. Vol", id='btn-g2-none', n_clicks=0, style=btn_style),
            ], style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '10px'}),
            dcc.Graph(id='global-graph-2', style={'height': '400px'}, config={'displayModeBar': False}),
            dcc.Store(id='store-g2', data='portfolio')
        ], style={**card_style, 'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ], style={'display': 'flex', 'justifyContent': 'space-between'}),

    # PASEK POSTĘPU
    html.Div([
        html.Div([
            html.Span("Postęp Treningu: ", style={'color': '#888'}),
            html.Span(id='progress-text', style={'color': COLORS['accent'], 'fontWeight': 'bold'})
        ], style={'marginBottom': '5px'}),
        html.Div([
            html.Div(id='progress-bar', style={'width': '0%', 'height': '100%', 'backgroundColor': COLORS['success'],
                                               'transition': 'width 0.5s'})
        ], style={'width': '100%', 'height': '10px', 'backgroundColor': '#333', 'borderRadius': '5px',
                  'overflow': 'hidden'})
    ], style={'margin': '20px 15px 40px 15px'}),

    html.Hr(style={'borderColor': '#333', 'marginBottom': '30px'})
])

# --- GŁÓWNY LAYOUT ---
app.layout = html.Div(
    style={'backgroundColor': COLORS['bg'], 'color': COLORS['text'], 'fontFamily': 'Roboto, sans-serif',
           'minHeight': '100vh', 'padding': '20px'},
    children=[
        html.H1(id='main-title', children="Monitor Treningu Agenta RL",
                style={'textAlign': 'center', 'color': '#ffffff', 'marginBottom': '10px', 'fontSize': '36px',
                       'fontWeight': 'bold'}),

        dcc.Store(id='activity-store', data={'last_epoch': -1, 'last_timestamp': time.time()}),
        global_section,

        # ORYGINALNY GRID NA DOLE
        html.Div([
            html.Div([
                html.Button("🔄 AUTO", id='btn-auto', n_clicks=0, style=btn_active_style),
                html.Button("🥧 Decyzje", id='btn-pie', n_clicks=0, style=btn_style),
                html.Button("💰 Portfel", id='btn-port', n_clicks=0, style=btn_style),
                html.Button("📈 Nagrody", id='btn-rew', n_clicks=0, style=btn_style),
            ], style={'textAlign': 'center', 'marginBottom': '20px'}),

            dcc.Store(id='view-mode-store', data='auto'),

            html.Div([
                html.Span(id='global-title',
                          style={'fontSize': '20px', 'color': '#00d4ff', 'fontWeight': 'bold', 'marginRight': '20px'}),
                html.Span(id='process-count', style={'color': '#888', 'fontSize': '16px'})
            ], style={'textAlign': 'center', 'marginBottom': '20px'}),

            html.Div(id='graphs-container', style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'}),
        ]),

        dcc.Interval(id='interval-component', interval=UPDATE_INTERVAL_MS, n_intervals=0)
    ]
)


# --- LOGIKA GENEROWANIA WYKRESÓW GLOBALNYCH ---
def generate_global_fig(mode, full_data):
    fig = go.Figure()

    # Style layoutu
    layout_args = dict(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=20, t=40, b=40)
    )
    title_defaults = dict(x=0.5, xanchor='center', font=dict(size=16))

    if not full_data or not full_data.get('history'):
        fig.update_layout(title=dict(text="Oczekiwanie na dane...", **title_defaults), **layout_args)
        return fig

    hist = full_data['history']
    config = full_data.get('config', {})

    # Używamy sumy epok dla osi X lub numeru porządkowego
    # epochs = [h.get('epoch_sum', i) for i, h in enumerate(hist, 1)]
    epochs = list(range(1, len(hist) + 1))
    max_x = config.get('global_epochs', 100)

    # --- 1. REWARD SUMMARY (Button: Reward Avg) ---
    if mode == 'reward':
        avg = [h['reward']['avg_end_r'] for h in hist]
        mn = [h['reward']['min_end_r'] for h in hist]
        mx = [h['reward']['max_end_r'] for h in hist]

        fig.add_trace(go.Scatter(
            x=epochs + epochs[::-1], y=mx + mn[::-1],
            fill='toself', fillcolor='rgba(255, 145, 0, 0.15)',
            line=dict(color='rgba(0,0,0,0)'), name='Range'
        ))
        fig.add_trace(go.Scatter(
            x=epochs, y=avg, mode='lines+markers',
            marker=dict(size=[0] * (len(avg) - 1) + [6], color='white'),
            line=dict(color=COLORS['warning'], width=2), name='Avg End Reward'
        ))
        fig.update_layout(
            title=dict(text="Wykres średniej końcowej wartości nagrody", **title_defaults),
            xaxis_title="Global Epoch", yaxis_title="Reward", **layout_args,
            xaxis=dict(range=[1, max_x]),
            legend=dict(
                orientation="h",  # Pozioma
                yanchor="top",
                y=-0.3,  # Pod wykresem
                xanchor="center",
                x=0.5,
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=15)
            ),
        )

    # --- 2. PORTFOLIO SUMMARY (Button: Portfolio Avg) ---
    elif mode == 'portfolio':
        avg = [h['portfolio']['avg_end_p'] for h in hist]
        mn = [h['portfolio']['min_end_p'] for h in hist]
        mx = [h['portfolio']['max_end_p'] for h in hist]

        fig.add_trace(go.Scatter(
            x=epochs + epochs[::-1], y=mx + mn[::-1],
            fill='toself', fillcolor='rgba(0, 212, 255, 0.15)',
            line=dict(color='rgba(0,0,0,0)'), name='Range'
        ))
        fig.add_trace(go.Scatter(
            x=epochs, y=avg, mode='lines+markers',
            marker=dict(size=[0] * (len(avg) - 1) + [6], color='white'),
            line=dict(color=COLORS['accent'], width=2), name='Avg End Portfolio'
        ))
        fig.update_layout(
            title=dict(text="Wykres średniej końcowej wartości portfela", **title_defaults),
            xaxis_title="Global Epoch", yaxis_title="Value ($)", **layout_args,
            xaxis=dict(range=[1, max_x]),
            legend=dict(
                orientation="h",  # Pozioma
                yanchor="top",
                y=-0.3,  # Pod wykresem
                xanchor="center",
                x=0.5,
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=15)
            ),
        )

    # --- 3. PORTFOLIO VOLATILITY (Button: Heatmap -> Port. Vol) ---
    elif mode == 'heatmap':
        max_max = [h['portfolio']['max_max_p'] for h in hist]
        min_min = [h['portfolio']['min_min_p'] for h in hist]
        max_mean = [h['portfolio']['max_mean_p'] for h in hist]
        min_mean = [h['portfolio']['min_mean_p'] for h in hist]

        fig.add_trace(go.Scatter(
            x=epochs + epochs[::-1], y=max_max + min_min[::-1],
            fill='toself', fillcolor='rgba(0, 89, 255, 0.2)',
            line=dict(color='rgba(0,0,0,0)'), name='Global Extr.'
        ))
        fig.add_trace(go.Scatter(
            x=epochs + epochs[::-1], y=max_mean + min_mean[::-1],
            fill='toself', fillcolor='rgba(0, 212, 255, 0.4)',
            line=dict(color='rgba(0,0,0,0)'), name='Mean Range'
        ))
        fig.update_layout(
            title=dict(text="Wykres zmienności portfela", **title_defaults),
            xaxis_title="Global Epoch", yaxis_title="Value ($)", **layout_args,
            xaxis=dict(range=[1, max_x]),
            legend=dict(
                orientation="h",  # Pozioma
                yanchor="top",
                y=-0.3,  # Pod wykresem
                xanchor="center",
                x=0.5,
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=15)
            ),
        )

    # --- 4. TRANSACTIONS HISTOGRAM (Button: Histogram) ---
    elif mode == 'histogram':
        buys, sells = [], []
        for h in hist:
            tot = h['avg_transactions']
            b, s = h['avg_buys'], h['avg_sells']
            if (b + s) > tot and (b + s) > 0:  # Korekta
                diff = (b + s) - tot
                b -= diff * (b / (b + s))
                s -= diff * (s / (b + s))
            buys.append(b)
            sells.append(s)

        fig.add_trace(go.Bar(x=epochs, y=buys, name='Buys', marker_color=COLORS['success']))
        fig.add_trace(go.Bar(x=epochs, y=sells, name='Sells', marker_color=COLORS['danger']))

        ep_len = config.get('episode_length', 1000)
        fig.update_layout(
            title=dict(text="Wykres wykonanych transakcji", **title_defaults),
            barmode='stack', yaxis_range=[0, ep_len], yaxis_title="Transactions",xaxis_title="Global Epoch", xaxis=dict(range=[1, max_x]), **layout_args,
            legend=dict(
                orientation="h",  # Pozioma
                yanchor="top",
                y=-0.3,  # Pod wykresem
                xanchor="center",
                x=0.5,
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=15)
            ),
        )

    # --- 5. REWARD VOLATILITY (Button: Empty -> Rew. Vol) ---
    elif mode == 'empty':
        max_max = [h['reward']['max_max_r'] for h in hist]
        min_min = [h['reward']['min_min_r'] for h in hist]
        max_mean = [h['reward']['max_mean_r'] for h in hist]
        min_mean = [h['reward']['min_mean_r'] for h in hist]

        fig.add_trace(go.Scatter(
            x=epochs + epochs[::-1], y=max_max + min_min[::-1],
            fill='toself', fillcolor='rgba(200, 50, 0, 0.2)',
            line=dict(color='rgba(0,0,0,0)'), name='Global Extr.'
        ))
        fig.add_trace(go.Scatter(
            x=epochs + epochs[::-1], y=max_mean + min_mean[::-1],
            fill='toself', fillcolor='rgba(255, 145, 0, 0.4)',
            line=dict(color='rgba(0,0,0,0)'), name='Mean Range'
        ))
        fig.update_layout(
            title=dict(text="Wykres zmienności nagrody", **title_defaults),
            xaxis_title="Global Epoch",yaxis_title="Reward", **layout_args,
            xaxis=dict(range=[1, max_x]),
            legend=dict(
                orientation="h",  # Pozioma
                yanchor="top",
                y=-0.3,  # Pod wykresem
                xanchor="center",
                x=0.5,
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=15)
            ),
        )

    return fig


# --- CALLBACKI (Bez zmian w logice, tylko podpięcie nowej funkcji rysowania) ---

# Status aktywności
@app.callback(
    [Output('main-title', 'children'), Output('activity-store', 'data')],
    [Input('interval-component', 'n_intervals')],
    [State('activity-store', 'data')]
)
def update_activity_status(n, activity_data):
    full_data = get_full_training_data()
    hist = full_data.get('history', []) if full_data else []

    last_ep = activity_data.get('last_epoch', -1)
    last_time = activity_data.get('last_timestamp', time.time())
    current_time = time.time()

    current_ep_sum = hist[-1]['epoch_sum'] if hist else 0

    if current_ep_sum > last_ep:
        new_last_time = current_time
        new_last_ep = current_ep_sum
    else:
        new_last_time = last_time
        new_last_ep = last_ep

    is_active = (current_time - new_last_time) < 300  # 5 min timeout
    status_text = "🟢 (Aktywny)" if is_active else "🔴 (Nieaktywny)"
    if not hist: status_text = "🟡 (Oczekiwanie)"

    title = ["Monitor Treningu Agenta RL ", html.Span(status_text, style={'fontSize': '20px', 'color': '#ccc'})]
    return title, {'last_epoch': new_last_ep, 'last_timestamp': new_last_time}


# Panel Lewy
@app.callback(
    [Output('global-graph-1', 'figure'), Output('store-g1', 'data'),
     Output('btn-g1-rew', 'style'), Output('btn-g1-port', 'style'), Output('btn-g1-heat', 'style'),
     Output('btn-g1-hist', 'style'), Output('btn-g1-none', 'style')],
    [Input('interval-component', 'n_intervals'),
     Input('btn-g1-rew', 'n_clicks'), Input('btn-g1-port', 'n_clicks'), Input('btn-g1-heat', 'n_clicks'),
     Input('btn-g1-hist', 'n_clicks'), Input('btn-g1-none', 'n_clicks')],
    [State('store-g1', 'data')]
)
def update_g1(n, b1, b2, b3, b4, b5, current):
    ctx = dash.callback_context
    if ctx.triggered and 'btn' in ctx.triggered[0]['prop_id']:
        bid = ctx.triggered[0]['prop_id']
        if 'rew' in bid:
            current = 'reward'
        elif 'port' in bid:
            current = 'portfolio'
        elif 'heat' in bid:
            current = 'heatmap'
        elif 'hist' in bid:
            current = 'histogram'
        elif 'none' in bid:
            current = 'empty'

    styles = [btn_style] * 5
    modes = ['reward', 'portfolio', 'heatmap', 'histogram', 'empty']
    if current in modes: styles[modes.index(current)] = btn_active_style

    return generate_global_fig(current, get_full_training_data()), current, *styles


# Panel Prawy
@app.callback(
    [Output('global-graph-2', 'figure'), Output('store-g2', 'data'),
     Output('btn-g2-port', 'style'), Output('btn-g2-rew', 'style'), Output('btn-g2-heat', 'style'),
     Output('btn-g2-hist', 'style'), Output('btn-g2-none', 'style')],
    [Input('interval-component', 'n_intervals'),
     Input('btn-g2-port', 'n_clicks'), Input('btn-g2-rew', 'n_clicks'), Input('btn-g2-heat', 'n_clicks'),
     Input('btn-g2-hist', 'n_clicks'), Input('btn-g2-none', 'n_clicks')],
    [State('store-g2', 'data')]
)
def update_g2(n, b1, b2, b3, b4, b5, current):
    ctx = dash.callback_context
    if ctx.triggered and 'btn' in ctx.triggered[0]['prop_id']:
        bid = ctx.triggered[0]['prop_id']
        if 'port' in bid:
            current = 'portfolio'
        elif 'rew' in bid:
            current = 'reward'
        elif 'heat' in bid:
            current = 'heatmap'
        elif 'hist' in bid:
            current = 'histogram'
        elif 'none' in bid:
            current = 'empty'

    styles = [btn_style] * 5
    modes = ['portfolio', 'reward', 'heatmap', 'histogram', 'empty']
    if current in modes: styles[modes.index(current)] = btn_active_style

    return generate_global_fig(current, get_full_training_data()), current, *styles


# Pasek Postępu
@app.callback([Output('progress-bar', 'style'), Output('progress-text', 'children')],
              [Input('interval-component', 'n_intervals')])
def update_prog(n):
    full_data = get_full_training_data()
    max_eps = full_data.get('config', {}).get('max_episodes', TARGET_EPOCHS) if full_data else TARGET_EPOCHS
    num_cpu = full_data.get('config', {}).get('num_processes', TARGET_EPOCHS) if full_data else TARGET_EPOCHS
    current = len(full_data.get('history', []))*num_cpu if full_data else 0
    pct = min(100, (current / max_eps) * 100)
    return {'width': f'{pct}%', 'height': '100%', 'backgroundColor': COLORS['success'],
            'transition': 'width 0.5s'}, f"{current}/{max_eps} ({pct:.1f}%)"


# Grid - Zmiana trybu
@app.callback(
    [Output('view-mode-store', 'data'),
     Output('btn-auto', 'style'), Output('btn-pie', 'style'),
     Output('btn-port', 'style'), Output('btn-rew', 'style')],
    [Input('btn-auto', 'n_clicks'), Input('btn-pie', 'n_clicks'),
     Input('btn-port', 'n_clicks'), Input('btn-rew', 'n_clicks')],
    [State('view-mode-store', 'data')]
)
def change_mode(b1, b2, b3, b4, current_mode):
    ctx = dash.callback_context
    if not ctx.triggered: return 'auto', btn_active_style, btn_style, btn_style, btn_style
    bid = ctx.triggered[0]['prop_id'].split('.')[0]
    new = 'auto'
    if bid == 'btn-pie':
        new = 0
    elif bid == 'btn-port':
        new = 1
    elif bid == 'btn-rew':
        new = 2
    s = [btn_style] * 4
    if new == 'auto':
        s[0] = btn_active_style
    else:
        s[new + 1] = btn_active_style
    return new, *s



# --- CALLBACK: Grid - Aktualizacja wykresów procesów ---
@app.callback(
    [Output('graphs-container', 'children'), Output('global-title', 'children'), Output('process-count', 'children')],
    [Input('interval-component', 'n_intervals'), Input('view-mode-store', 'data')]
)
def update_grid(n_intervals, view_mode):
    data_list = get_process_data()
    if not data_list: return [html.Div("Oczekiwanie...", style={'color': 'gray'})], "Brak Danych", ""

    current_view = n_intervals % 3 if view_mode == 'auto' else view_mode

    global_title_text = "Widok: Decyzje"
    if current_view == 1:
        global_title_text = "Widok: Portfel"
    elif current_view == 2:
        global_title_text = "Widok: Nagrody (skumulowane)"

    graphs_html = []
    for p in data_list:
        fig = go.Figure()

        metrics = p.get('metrics', {'buys': 0, 'sells': 0, 'holds': 0})
        port_data = p.get('portfolio', {})
        rew_data = p.get('reward', {})

        p_id = p.get('process_id', '?')
        epoch = p.get('epoch', '?')

        # --- View 0: Pie (Bez zmian) ---
        if current_view == 0:
            v = [metrics['buys'], metrics['sells'], metrics['holds']]
            if sum(v) == 0: v = [0, 0, 1]
            fig.add_trace(go.Pie(labels=['Buy', 'Sell', 'Hold'], values=v, hole=0.5,
                                 marker=dict(colors=[COLORS['success'], COLORS['danger'], '#757575']),
                                 textinfo='percent'))

        # --- View 1: Portfolio ---
        elif current_view == 1:
            y = port_data.get('portfolio_values', [])
            x = list(range(len(y)))

            # 1. Główna linia
            fig.add_trace(go.Scatter(
                x=x, y=y, mode='lines',
                fill='tozeroy', fillcolor='rgba(0, 212, 255, 0.1)',
                line=dict(color=COLORS['accent'], width=2),
                name='Current Val'  # <-- Podpis głównej linii
            ))

            # 2. Linie pomocnicze jako Traces (żeby były w legendzie)
            mn = port_data.get('min_p')
            mx = port_data.get('max_p')
            avg = port_data.get('mean_p')

            # Aby narysować linię poziomą "Scatterem", potrzebujemy punktu początkowego i końcowego osi X
            x_range = [x[0], x[-1]] if x else [0, 0]

            if mx:
                fig.add_trace(go.Scatter(
                    x=x_range, y=[mx, mx], mode='lines',
                    line=dict(color=COLORS['success'], dash='dot', width=1),
                    name='Max'
                ))
            if mn:
                fig.add_trace(go.Scatter(
                    x=x_range, y=[mn, mn], mode='lines',
                    line=dict(color=COLORS['danger'], dash='dot', width=1),
                    name='Min'
                ))
            if avg:
                fig.add_trace(go.Scatter(
                    x=x_range, y=[avg, avg], mode='lines',
                    line=dict(color='#bbbbbb', dash='dash', width=1),  # Szary
                    name='Mean'
                ))

            # Skalowanie
            if y:
                margin = (max(y) - min(y)) * 0.1 if len(y) > 1 else 100
                if margin == 0: margin = 100
                fig.update_layout(yaxis=dict(range=[min(y) - margin, max(y) + margin]))

        # --- View 2: Reward ---
        elif current_view == 2:
            y = rew_data.get('reward_values', [])
            x = list(range(len(y)))

            # 1. Główna linia
            fig.add_trace(go.Scatter(
                x=x, y=y, mode='lines',
                line=dict(color=COLORS['warning'], width=2),
                name='Reward'  # <-- Podpis głównej linii
            ))

            # Linia 0 (jako adnotacja, bo to punkt odniesienia, nie musi być w legendzie)
            fig.add_hline(y=0, line_color="#444", line_width=1)

            # 2. Linie pomocnicze
            mn = rew_data.get('min_r')
            mx = rew_data.get('max_r')
            avg = rew_data.get('mean_r')

            x_range = [x[0], x[-1]] if x else [0, 0]

            if mx:
                fig.add_trace(go.Scatter(
                    x=x_range, y=[mx, mx], mode='lines',
                    line=dict(color=COLORS['success'], dash='dot', width=1),
                    name='Max'
                ))
            if mn:
                fig.add_trace(go.Scatter(
                    x=x_range, y=[mn, mn], mode='lines',
                    line=dict(color=COLORS['danger'], dash='dot', width=1),
                    name='Min'
                ))
            if avg:
                fig.add_trace(go.Scatter(
                    x=x_range, y=[avg, avg], mode='lines',
                    line=dict(color='#bbbbbb', dash='dash', width=1),
                    name='Mean'
                ))

        # --- Konfiguracja Layoutu (Legenda pod wykresem) ---
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12, color="#ccc", family='Roboto, sans-serif'),
            margin=dict(l=20, r=20, t=30, b=50),  # b=50 daje miejsce na legendę
            xaxis=dict(visible=False), yaxis=dict(showgrid=True, gridcolor='#333'),
            title=dict(text=f"Proc: {p_id} | Epoka: {epoch}", font=dict(size=12, color="white"), x=0.5,
                       xanchor='center'),

            # USTAWIENIA LEGENDY
            showlegend=True,
            legend=dict(
                orientation="h",  # Pozioma
                yanchor="top",
                y=-0.1,  # Pod wykresem
                xanchor="center",
                x=0.5,
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=12)
            ),
            height=300
        )
        card = html.Div([dcc.Graph(figure=fig, config={'displayModeBar': False})], style=card_style)
        graphs_html.append(card)

    return graphs_html, global_title_text, f"Aktywne procesy: {len(data_list)}"


if __name__ == '__main__':
    app.run(debug=False, port=8050)
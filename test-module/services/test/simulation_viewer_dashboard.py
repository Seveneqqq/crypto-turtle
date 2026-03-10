import dash
from dash import dcc, html, ctx
from dash.dependencies import Input, Output, State, ALL, MATCH
import plotly.graph_objs as go
import plotly.io as pio
import json
import gzip
import os
import glob
import re

# --- KONFIGURACJA ŚCIEŻEK ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUTPUT_DIR = os.path.join(BASE_DIR, "shared", "simulation", "outputs")

# --- DEFINICJE MOTYWÓW ---
THEMES = {
    'dark': {
        'bg': '#111111',
        'sidebar': '#0a0a0a',
        'card': '#1e1e1e',
        'text': '#e0e0e0',
        'subtext': '#888888',
        'border': '#333',
        'chart_template': 'plotly_dark',
        'success': '#00e676',
    },
    'light': {
        'bg': '#f4f4f9',
        'sidebar': '#ffffff',
        'card': '#ffffff',
        'text': '#333333',
        'subtext': '#666666',
        'border': '#e0e0e0',
        'chart_template': 'plotly_white',
        'success': '#00c853',
    }
}

# Stałe kolory akcentów (niezależne od motywu)
ACCENTS = {
    'blue': '#448aff',
    'red': '#ff5252',
    'orange': '#ffab40'
}

pio.templates.default = "plotly_dark"

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Simulation Viewer"

# --- CUSTOM CSS ---
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
        <style>
            body {
                font-family: 'Roboto', sans-serif !important;
                margin: 0;
                overflow-x: hidden;
                box-sizing: border-box;
                transition: background-color 0.3s ease;
            }
            *, *:before, *:after {
                box-sizing: inherit;
            }

            /* Custom Scrollbar */
            ::-webkit-scrollbar { width: 8px; height: 8px; }
            ::-webkit-scrollbar-track { background: rgba(0,0,0,0.1); }
            ::-webkit-scrollbar-thumb { background: rgba(128,128,128,0.5); border-radius: 4px; }
            ::-webkit-scrollbar-thumb:hover { background: rgba(128,128,128,0.8); }

            /* Tabs Container */
            .custom-tabs-container {
                margin-top: 20px;
                width: 100%;
            }

            /* Grid layout for Metrics */
            .responsive-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
                width: 100%;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# --- BAZOWE STYLE ---
sidebar_base_style = {
    'position': 'fixed', 'top': 0, 'left': 0, 'bottom': 0, 'width': '20rem',
    'padding': '2rem 1rem', 'overflowY': 'auto',
    'zIndex': 100,
    'transition': 'background-color 0.3s, border-right 0.3s'
}

content_base_style = {
    'marginLeft': '20rem',
    'padding': '40px',
    'minHeight': '100vh',
    'overflowX': 'hidden',
    'transition': 'background-color 0.3s, color 0.3s'
}

file_btn_base_style = {
    'width': '100%', 'textAlign': 'left', 'padding': '12px', 'marginBottom': '8px',
    'backgroundColor': 'transparent',
    'cursor': 'pointer', 'borderRadius': '6px', 'fontSize': '13px', 'transition': '0.2s',
    'fontFamily': 'Roboto, sans-serif'
}

metric_card_base_style = {
    'borderRadius': '8px', 'padding': '15px',
    'textAlign': 'center',
    'width': '100%',
    'boxSizing': 'border-box',
    'transition': 'background-color 0.3s, border 0.3s, box-shadow 0.3s'
}

section_header_base_style = {
    'paddingBottom': '5px',
    'marginBottom': '15px',
    'marginTop': '30px',
    'fontWeight': 'bold',
    'fontSize': '18px',
    'letterSpacing': '1px',
    'width': '100%'
}

close_btn_style = {
    'position': 'absolute',
    'top': '0px',
    'right': '0px',
    'backgroundColor': 'rgba(255, 82, 82, 0.1)',
    'border': '1px solid #ff5252',
    'color': '#ff5252',
    'padding': '6px 12px',
    'borderRadius': '4px',
    'cursor': 'pointer',
    'fontWeight': 'bold',
    'fontSize': '12px',
    'zIndex': 10,
    'transition': '0.2s'
}


# --- FUNKCJE POMOCNICZE ---

def clean_filename(filename):
    return filename.replace(".json.gz", "").replace(".json", "")


def parse_simulation_name(filename):
    base = clean_filename(filename)
    try:
        date_match = re.search(r'(\d{2})_(\d{2})', base)
        ver_match = re.search(r'_v(\d+)', base)
        day = int(date_match.group(1)) if date_match else 0
        month = int(date_match.group(2)) if date_match else 0
        version = int(ver_match.group(1)) if ver_match else 0
        file_time = os.path.getmtime(os.path.join(OUTPUT_DIR, filename))
        return (month, day, version, file_time)
    except:
        return (0, 0, 0, os.path.getmtime(os.path.join(OUTPUT_DIR, filename)))


def get_simulation_files():
    if not os.path.exists(OUTPUT_DIR): return []
    files = glob.glob(os.path.join(OUTPUT_DIR, "*.json.gz"))
    files = [os.path.basename(f) for f in files]
    files.sort(key=parse_simulation_name, reverse=True)
    return files


def load_sim_data(filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    try:
        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None


def format_value_for_card(value):
    if isinstance(value, float): return f"{value:.4f}"
    if isinstance(value, list):
        filenames = [os.path.basename(str(v)) for v in value]
        if not filenames: return "-"
        if len(filenames) > 3: return f"{len(filenames)} files"
        return ", ".join(filenames)
    if isinstance(value, dict):
        items = [f"{k}: {os.path.basename(str(v))}" for k, v in value.items()]
        return html.Div([html.Div(i, style={'fontSize': '12px', 'marginBottom': '2px'}) for i in items])
    return str(value)


def create_info_cards(info_dict, theme_name, text_color=None):
    cards = []
    thm = THEMES[theme_name]
    if text_color is None: text_color = thm['text']

    if not info_dict: return [html.Div("Brak danych", style={'color': thm['subtext'], 'padding': '10px'})]

    # Styl początkowy
    style = {
        **metric_card_base_style,
        'backgroundColor': thm['card'],
        'border': f'1px solid {thm["border"]}',
        'boxShadow': '0 4px 6px rgba(0,0,0,0.1)' if theme_name == 'light' else '0 4px 6px rgba(0,0,0,0.3)'
    }

    for key, value in info_dict.items():
        display_value = format_value_for_card(value)
        display_key = key.replace("_", " ").title()

        # ID pozwala na update stylu w callbacku
        card = html.Div([
            html.Div(display_value,
                     style={'color': text_color, 'fontSize': '16px', 'fontWeight': 'bold', 'margin': '0 0 5px 0',
                            'wordBreak': 'break-word'}),
            html.Span(display_key, style={'color': thm['subtext'], 'fontSize': '11px', 'textTransform': 'uppercase'})
        ], style=style, id={'type': 'metric-card', 'index': key + str(os.urandom(4))})
        cards.append(card)
    return cards


def generate_comparison_chart(data, theme_name='dark'):
    fig = go.Figure()
    all_values = []
    thm = THEMES[theme_name]

    # 1. HODL
    hodl_data = data.get('hodl', {}).get('crypto_ref_values', {})
    for symbol, values in hodl_data.items():
        all_values.extend(values)
        fig.add_trace(go.Scatter(
            y=values, mode='lines', name=f'HODL {symbol}',
            line=dict(color=ACCENTS['orange'], width=1.5, dash='dash'), opacity=0.7
        ))

    # 2. AGENT RL
    if 'agent' in data:
        agent_vals = data['agent'].get('portfolio_values', [])
        all_values.extend(agent_vals)
        fig.add_trace(go.Scatter(
            y=agent_vals, mode='lines', name='Agent RL',
            line=dict(color=ACCENTS['red'], width=2.5),
            fill='tozeroy', fillcolor='rgba(255, 82, 82, 0.05)'
        ))

    # 3. MODEL ML
    if 'model' in data:
        model_vals = data['model'].get('portfolio_values', [])
        all_values.extend(model_vals)
        fig.add_trace(go.Scatter(
            y=model_vals, mode='lines', name='Model ML',
            line=dict(color=ACCENTS['blue'], width=2.5),
        ))

    y_range = None
    if all_values:
        min_v, max_v = min(all_values), max(all_values)
        y_range = [min_v - 100, max_v + 100]

    fig.update_layout(
        template=thm['chart_template'],
        title=dict(text="Portfolio Value vs Benchmark", font=dict(family='Roboto', size=20, color=thm['text']), x=0.5,
                   xanchor='center'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=thm['text'], family='Roboto'),
        xaxis=dict(showgrid=True, gridcolor=thm['border'], title="Timesteps", title_font=dict(size=14),
                   tickfont=dict(size=14)),
        yaxis=dict(showgrid=True, gridcolor=thm['border'], title="Value ($)", range=y_range, title_font=dict(size=14),
                   tickfont=dict(size=14)),
        legend=dict(
            orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02,
            bgcolor='rgba(0,0,0,0)', font=dict(size=14)
        ),
        margin=dict(l=60, r=20, t=60, b=40),
        height=550
    )
    return fig


# --- LAYOUT ---

app.layout = html.Div([
    dcc.Store(id='open-files-store', data=[]),
    dcc.Store(id='theme-store', data='dark'),
    dcc.Interval(id='auto-refresh', interval=30 * 1000, n_intervals=0),

    # Przycisk zmiany motywu
    html.Button("☀/🌙", id='theme-toggle', n_clicks=0, style={
        'position': 'fixed', 'top': '20px', 'right': '20px', 'zIndex': 1000,
        'padding': '10px', 'borderRadius': '5px', 'cursor': 'pointer',
        'border': 'none', 'backgroundColor': 'rgba(128,128,128,0.2)', 'color': 'inherit'
    }),

    # PANEL BOCZNY
    html.Div(id='sidebar', children=[
        html.H2("Symulacje", id='sidebar-title',
                style={'marginBottom': '20px', 'textAlign': 'center', 'letterSpacing': '2px', 'fontSize': '22px'}),
        html.Hr(id='sidebar-hr', style={'margin': '20px 0'}),
        html.Div(id='file-list-container')
    ], style=sidebar_base_style),

    # GŁÓWNY PANEL
    html.Div(id='content', children=[
        html.H1("Analiza Wyników", id='main-title', style={
            'textAlign': 'center',
            'paddingBottom': '15px',
            'marginBottom': '25px',
            'letterSpacing': '2px',
        }),

        html.Div([
            dcc.Tabs(id='simulation-tabs', value='', className='custom-tabs', children=[])
        ], className='custom-tabs-container')

    ], style=content_base_style)
])


# --- CALLBACKI ---

# 1. Przełączanie Motywu (Store)
@app.callback(
    Output('theme-store', 'data'),
    [Input('theme-toggle', 'n_clicks')],
    [State('theme-store', 'data')]
)
def toggle_theme(n, current):
    if n > 0:
        return 'light' if current == 'dark' else 'dark'
    return 'dark'


# 2. Aktualizacja Głównego Stylu (Sidebar, Tło, Tytuły)
@app.callback(
    [Output('sidebar', 'style'),
     Output('content', 'style'),
     Output('sidebar-title', 'style'),
     Output('main-title', 'style'),
     Output('sidebar-hr', 'style')],
    [Input('theme-store', 'data')],
    [State('sidebar', 'style'), State('content', 'style')]
)
def update_global_styles(theme_name, sidebar_s, content_s):
    thm = THEMES[theme_name]

    # Sidebar
    new_sidebar = sidebar_s.copy()
    new_sidebar['backgroundColor'] = thm['sidebar']
    new_sidebar['borderRight'] = f"1px solid {thm['border']}"

    # Content
    new_content = content_s.copy()
    new_content['backgroundColor'] = thm['bg']
    new_content['color'] = thm['text']

    # Titles
    title_style = {'color': thm['text'], 'marginBottom': '20px', 'textAlign': 'center', 'letterSpacing': '2px',
                   'fontSize': '22px'}
    main_title_style = {'color': thm['text'], 'textAlign': 'center', 'borderBottom': f"1px solid {thm['border']}",
                        'paddingBottom': '15px', 'marginBottom': '25px', 'letterSpacing': '2px'}
    hr_style = {'borderColor': thm['border'], 'margin': '20px 0'}

    return new_sidebar, new_content, title_style, main_title_style, hr_style


# 3. Aktualizacja Listy Plików (Styl przycisków + Aktywny plik)
@app.callback(
    Output('file-list-container', 'children'),
    [Input('auto-refresh', 'n_intervals'),
     Input('open-files-store', 'data'),
     Input('theme-store', 'data')]
)
def update_file_list(n, open_files, theme_name):
    files = get_simulation_files()
    thm = THEMES[theme_name]

    if not files:
        return html.Div("Brak plików", style={'color': thm['subtext'], 'textAlign': 'center', 'marginTop': '20px'})

    buttons = []
    for f in files:
        # Bazowy styl
        style = file_btn_base_style.copy()
        style['color'] = thm['text']
        style['border'] = f"1px solid {thm['border']}"

        # Styl aktywnego pliku (Niebieska ramka)
        if f in open_files:
            style['border'] = f"1px solid {ACCENTS['blue']}"
            style['borderLeft'] = f"4px solid {ACCENTS['blue']}"
            style['backgroundColor'] = 'rgba(68, 138, 255, 0.1)'
            style['fontWeight'] = 'bold'

        buttons.append(html.Button(
            f"📈 {clean_filename(f)}",
            id={'type': 'file-btn', 'index': f},
            n_clicks=0,
            style=style
        ))
    return buttons


def get_tab_styles(theme_name):
    """Zwraca style dla zakładek w zależności od motywu."""
    thm = THEMES[theme_name]

    # Styl nieaktywnej zakładki
    tab_style = {
        'backgroundColor': thm['card'],
        'color': thm['subtext'],
        'border': f'1px solid {thm["border"]}',
        'padding': '12px 20px',
        'fontFamily': 'Roboto, sans-serif',
        'width': 'auto',
        'flex': 'none',
        'marginRight': '5px',
        'cursor': 'pointer'
    }

    # Styl aktywnej zakładki
    selected_tab_style = {
        'backgroundColor': thm['bg'],  # Tło takie jak strony
        'color': thm['text'],
        'fontWeight': 'bold',
        'borderTop': f'2px solid {ACCENTS["blue"]}',
        'borderBottom': 'none',  # Brak dolnej krawędzi (połączenie z treścią)
        'borderLeft': f'1px solid {thm["border"]}',
        'borderRight': f'1px solid {thm["border"]}',
        'padding': '12px 20px',
        'fontFamily': 'Roboto, sans-serif',
        'width': 'auto',
        'flex': 'none',
        'marginRight': '5px',
        'cursor': 'pointer'
    }

    return tab_style, selected_tab_style
# 4. Otwieranie/Zamykanie Zakładek
@app.callback(
    [Output('open-files-store', 'data'),
     Output('simulation-tabs', 'children'),
     Output('simulation-tabs', 'value')],
    [Input({'type': 'file-btn', 'index': ALL}, 'n_clicks'),
     Input({'type': 'close-tab-btn', 'index': ALL}, 'n_clicks'),
     Input('theme-store', 'data')],
    [State('open-files-store', 'data'),
     State('simulation-tabs', 'children')]
)
def manage_tabs(file_clicks, close_clicks, theme_name, open_files, current_tabs):
    trigger = ctx.triggered_id
    thm = THEMES[theme_name]

    # Pobierz aktualne style dla motywu (dla zakładek)
    tab_style, selected_tab_style = get_tab_styles(theme_name)

    # --- SCENARIUSZ 1: ZMIANA MOTYWU (Aktualizacja istniejących zakładek) ---
    if trigger == 'theme-store' or (not trigger and current_tabs):
        updated_tabs = []
        for tab in current_tabs:
            tab['props']['style'] = tab_style
            tab['props']['selected_style'] = selected_tab_style
            updated_tabs.append(tab)
        return dash.no_update, updated_tabs, dash.no_update

    if not trigger: return dash.no_update

    # --- SCENARIUSZ 2: OTWIERANIE PLIKU ---
    if isinstance(trigger, dict) and trigger.get('type') == 'file-btn':
        if not any(c for c in file_clicks if c is not None and c > 0): return dash.no_update

        filename = trigger['index']
        if filename in open_files: return dash.no_update, dash.no_update, f'tab-{filename}'

        data = load_sim_data(filename)
        if data is None: return dash.no_update, dash.no_update, dash.no_update

        # Generowanie treści
        main_fig = generate_comparison_chart(data, theme_name)
        sim_files = data.get('files', {})
        sim_config = data.get('config', {})
        sim_description = data.get('config', {})
        metrics_dict = {}
        if 'agent' in data and 'metrics' in data['agent']:
            for k, v in data['agent']['metrics'].items(): metrics_dict[f"[RL] {k}"] = v
        if 'model' in data and 'metrics' in data['model']:
            for k, v in data['model']['metrics'].items(): metrics_dict[f"[ML] {k}"] = v

        titles = []
        if 'agent' in data: titles.append(html.Span(f"Agent RL", style={'color': ACCENTS['red'], 'fontWeight': 'bold',
                                                                       'marginRight': '30px', 'fontSize': '20px'}))
        if 'model' in data: titles.append(html.Span(f"Model ML", style={'color': ACCENTS['blue'], 'fontWeight': 'bold',
                                                                       'marginLeft': '30px', 'fontSize': '20px'}))

        title_container = html.Div(titles, style={'textAlign': 'center', 'marginBottom': '30px', 'marginTop': '10px',
                                                  'paddingTop': '5px'})

        # WYKRES
        chart_section = html.Div([
            html.Div([
                dcc.Graph(
                    id={'type': 'graph', 'index': filename},
                    figure=main_fig,
                    style={'height': '550px', 'width': '100%'},
                    config={'responsive': True, 'displayModeBar': False}
                )
            ], id={'type': 'chart-container', 'index': filename},
                style={
                    'backgroundColor': thm['card'],
                    'padding': '0px',
                    'borderRadius': '10px',
                    'width': '100%',
                    'overflow': 'hidden',
                    'border': f'1px solid {thm["border"]}'
                })
        ], style={'width': '100%', 'marginBottom': '40px'})  # <--- DODANO MARGINES DOLNY

        # Styl nagłówków
        h_style = section_header_base_style.copy()
        h_style['color'] = thm['text']
        h_style['borderBottom'] = f'1px solid {thm["border"]}'

        # PLIKI
        files_section = html.Div([
            html.Div("📂 Simulation Files", style=h_style, id={'type': 'header', 'index': filename + '_files'}),
            html.Div(create_info_cards(sim_files, theme_name, thm['success']), className='responsive-grid')
        ], style={'marginBottom': '40px'})  # <--- DODANO MARGINES DOLNY

        # METRYKI
        metrics_section = html.Div([
            html.Div("📊 Simulation Metrics", style=h_style, id={'type': 'header', 'index': filename + '_metrics'}),
            html.Div(create_info_cards(metrics_dict, theme_name, ACCENTS['blue']), className='responsive-grid')
        ], style={'marginBottom': '40px'})  # <--- DODANO MARGINES DOLNY

        # KONFIGURACJA
        config_section = html.Div([
            html.Div("⚙️ Simulation Config", style=h_style, id={'type': 'header', 'index': filename + '_config'}),
            html.Div(create_info_cards(sim_config, theme_name, thm['text']), className='responsive-grid')
        ], style={'marginBottom': '50px'})  # <--- DODANO MARGINES DOLNY

        close_btn = html.Button("✕ Zamknij", id={'type': 'close-tab-btn', 'index': filename}, style=close_btn_style,
                                title="Zamknij zakładkę")

        tab_content = html.Div(
            [close_btn, html.Div(style={'clear': 'both'}), title_container, chart_section, metrics_section,
             files_section, config_section],
            style={'position': 'relative', 'marginTop': '10px', 'width': '100%'})

        new_tab = dcc.Tab(
            label=clean_filename(filename),
            value=f'tab-{filename}',
            children=tab_content,
            className='custom-tab',
            style=tab_style,
            selected_style=selected_tab_style
        )

        current_tabs.append(new_tab)
        open_files.append(filename)
        return open_files, current_tabs, f'tab-{filename}'

    # --- SCENARIUSZ 3: ZAMYKANIE PLIKU ---
    elif isinstance(trigger, dict) and trigger.get('type') == 'close-tab-btn':
        filename_to_close = trigger['index']
        if filename_to_close in open_files: open_files.remove(filename_to_close)
        new_tabs = [t for t in current_tabs if t['props']['value'] != f'tab-{filename_to_close}']
        next_tab = new_tabs[-1]['props']['value'] if new_tabs else ''
        return open_files, new_tabs, next_tab

    return dash.no_update


# 5. AKTUALIZACJA TREŚCI OTWARTYCH ZAKŁADEK PRZY ZMIANIE MOTYWU
@app.callback(
    [Output({'type': 'metric-card', 'index': ALL}, 'style'),
     Output({'type': 'chart-container', 'index': ALL}, 'style'),
     Output({'type': 'graph', 'index': ALL}, 'figure'),
     Output({'type': 'header', 'index': ALL}, 'style')],
    [Input('theme-store', 'data')],
    [State({'type': 'metric-card', 'index': ALL}, 'style'),
     State({'type': 'graph', 'index': ALL}, 'figure'),
     State({'type': 'header', 'index': ALL}, 'style')]
)
def update_tab_content_theme(theme_name, current_card_styles, current_figures, current_header_styles):
    thm = THEMES[theme_name]

    # 1. Cards
    new_card_styles = []
    for s in current_card_styles:
        new_s = s.copy()
        new_s['backgroundColor'] = thm['card']
        new_s['border'] = f"1px solid {thm['border']}"
        new_s['boxShadow'] = '0 4px 6px rgba(0,0,0,0.1)' if theme_name == 'light' else '0 4px 6px rgba(0,0,0,0.3)'
        new_card_styles.append(new_s)

    # 2. Charts Containers & Figures
    new_container_styles = []
    new_figures = []

    # Kontenery
    for _ in current_figures:
        new_container_styles.append({
            'backgroundColor': thm['card'],
            'padding': '0px',
            'borderRadius': '10px',
            'width': '100%',
            'overflow': 'hidden',
            'border': f'1px solid {thm["border"]}'
        })

    # Wykresy
    for fig_dict in current_figures:
        fig = go.Figure(fig_dict)
        fig.update_layout(
            template=thm['chart_template'],
            font=dict(color=thm['text']),
            xaxis=dict(gridcolor=thm['border']),
            yaxis=dict(gridcolor=thm['border']),
            title=dict(font=dict(color=thm['text']))
        )
        new_figures.append(fig)

    # 3. Headers
    new_header_styles = []
    for s in current_header_styles:
        new_s = s.copy()
        new_s['color'] = thm['text']
        new_s['borderBottom'] = f'1px solid {thm["border"]}'
        new_header_styles.append(new_s)

    return new_card_styles, new_container_styles, new_figures, new_header_styles


if __name__ == '__main__':
    app.run(debug=False, port=8052)
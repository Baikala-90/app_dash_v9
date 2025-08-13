
import os
import re
from datetime import datetime, timedelta

import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials

import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go

from dotenv import load_dotenv
import pytz

load_dotenv()
KST = pytz.timezone('Asia/Seoul')


def norm(s: str) -> str:
    return re.sub(r"\s+", "", (s or "")).strip()


def find_credentials_path():
    for p in [os.getenv("GOOGLE_APPLICATION_CREDENTIALS"), os.getenv("GSPREAD_CREDENTIALS"), "credentials.json", "service_account.json"]:
        if p and os.path.exists(p):
            print(f"[INFO] Using credentials: {p}")
            return p
    raise FileNotFoundError(
        "서비스 계정 JSON을 찾을 수 없습니다. (credentials.json / service_account.json)")


def open_sheet():
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    creds_file = find_credentials_path()
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_file, scope)
    client = gspread.authorize(creds)

    url = os.getenv("SPREADSHEET_URL", "").strip()
    if not url:
        raise EnvironmentError("SPREADSHEET_URL이 .env에 없습니다.")
    return client.open_by_url(url)


def load_data_from_gsheet():
    sh = open_sheet()
    daily_name = os.getenv("DAILY_SHEET_NAME", "일별 발주량 외").strip()
    monthly_name = os.getenv("MONTHLY_SHEET_NAME", "월별 발주량").strip()

    ws_d = sh.worksheet(daily_name)
    vals_d = ws_d.get_all_values()
    di = int(os.getenv("DAILY_HEADER_INDEX", "0"))
    headers_d = [norm(h) for h in vals_d[di]]
    df_daily_raw = pd.DataFrame(vals_d[di+1:], columns=headers_d)

    ws_m = sh.worksheet(monthly_name)
    vals_m = ws_m.get_all_values()
    mi = int(os.getenv("MONTHLY_HEADER_INDEX", "1"))
    headers_m = [norm(h) for h in vals_m[mi]]
    df_monthly_raw = pd.DataFrame(vals_m[mi+1:], columns=headers_m)

    return df_daily_raw, df_monthly_raw


def cleanse_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=['날짜', '총발주부수', '흑백페이지', '컬러페이지', 'date_only', '연도', '월'])

    d = df.copy()
    d = d.rename(columns={c: norm(c) for c in d.columns})

    col_date = next((c for c in d.columns if c in [
                    '날짜', '날짜(년월일)', '일자', '날']), d.columns[0])
    col_cnt = next((c for c in d.columns if c in [
                   '총발주종수', '총발주건수', '총발주건', '종수']), None)
    col_total = next((c for c in d.columns if c in [
                     '총발주부수', '총발주량', '총발주부', '총발주수', '총발주']), None)
    col_bw = next((c for c in d.columns if '흑백' in c), None)
    col_color = next((c for c in d.columns if ('컬러' in c or '칼라' in c)), None)

    d['__year_header__'] = d[col_date].astype(str).str.extract(
        r'(^\s*(\d{4})\s*년\s*$)', expand=True)[1]
    d['__year_header__'] = d['__year_header__'].ffill()

    raw = d[col_date].astype(str).str.strip().str.replace(
        r'\(.+?\)', '', regex=True).str.strip()
    full_pat = re.compile(r'^\d{4}[./-]\d{1,2}[./-]\d{1,2}$')
    partial_pat = re.compile(r'^\d{1,2}[./-]\d{1,2}$')

    def to_full_date(row):
        s = row['__raw__']
        if full_pat.match(s):
            return s.replace('-', '.')
        if partial_pat.match(s):
            y = row['__year_header__']
            if pd.notna(y):
                return f"{y}.{s.replace('-', '.')}"
        return None

    d['__raw__'] = raw
    d['__full__'] = d.apply(to_full_date, axis=1)
    d['날짜'] = pd.to_datetime(
        d['__full__'], errors='coerce', format='mixed', yearfirst=True)
    d = d.dropna(subset=['날짜']).copy()

    def is_blank(col):
        if col is None or col not in d.columns:
            return True
        return d[col].astype(str).str.strip().eq('').fillna(True)

    blanks = is_blank(col_cnt) & is_blank(
        col_total) & is_blank(col_bw) & is_blank(col_color)
    d = d.loc[~blanks].copy()

    for src, std in [(col_total, '총발주부수'), (col_bw, '흑백페이지'), (col_color, '컬러페이지')]:
        if src and src in d.columns:
            d[std] = pd.to_numeric(d[src].astype(str).str.replace(
                ',', '', regex=False), errors='coerce').fillna(0)
        else:
            d[std] = 0

    d['date_only'] = d['날짜'].dt.date
    d['연도'] = d['날짜'].dt.year
    d['월'] = d['날짜'].dt.month
    return d[['날짜', 'date_only', '연도', '월', '총발주부수', '흑백페이지', '컬러페이지']]


def cleanse_monthly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=['월DT', '발주량', '발주일수', '일평균발주량', '흑백출력량', '컬러출력량', '연도', '월번호'])

    d = df.copy()
    d = d.rename(columns={c: norm(c) for c in d.columns})
    month_col = '월' if '월' in d.columns else d.columns[0]

    rename_map = {}
    for c in d.columns:
        n = norm(c)
        if n == '발주량':
            rename_map[c] = '발주량'
        elif n == '발주일수':
            rename_map[c] = '발주일수'
        elif n == '일평균발주량':
            rename_map[c] = '일평균발주량'
        elif n == '흑백출력량':
            rename_map[c] = '흑백출력량'
        elif n in ['컬러출력량', '칼라출력량']:
            rename_map[c] = '컬러출력량'
    d = d.rename(columns=rename_map)

    d[month_col] = d[month_col].astype(str).str.replace(' ', '')
    d['월DT'] = pd.to_datetime(d[month_col], format='%Y년%m월', errors='coerce')
    d = d.dropna(subset=['월DT'])

    for c in ['발주량', '발주일수', '일평균발주량', '흑백출력량', '컬러출력량']:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c].astype(str).str.replace(
                ',', '', regex=False), errors='coerce').fillna(0)
        else:
            d[c] = 0

    d['연도'] = d['월DT'].dt.year
    d['월번호'] = d['월DT'].dt.month
    return d[['월DT', '연도', '월번호', '발주량', '발주일수', '일평균발주량', '흑백출력량', '컬러출력량']]


def last_5_business_days_upto_today(now_kst: datetime):
    dates = []
    d = now_kst.date()
    while len(dates) < 5:
        if d.weekday() < 5:
            dates.append(d)
        d -= timedelta(days=1)
    return list(reversed(dates))


WEEKDAY_KR = ['월', '화', '수', '목', '금', '토', '일']


def figure_weekly(df_daily: pd.DataFrame) -> go.Figure:
    now = datetime.now(KST)
    this_week_dates = last_5_business_days_upto_today(now)
    last_week_dates = [d - timedelta(days=7) for d in this_week_dates]

    m = df_daily.set_index('date_only')[
        '총발주부수'].to_dict() if not df_daily.empty else {}
    y_this = [m.get(d, 0) for d in this_week_dates]
    y_last = [m.get(d, 0) for d in last_week_dates]
    x_week = [WEEKDAY_KR[pd.Timestamp(d).weekday()] for d in this_week_dates]

    this_dates_str = [pd.Timestamp(d).strftime('%Y-%m-%d')
                      for d in this_week_dates]
    last_dates_str = [pd.Timestamp(d).strftime('%Y-%m-%d')
                      for d in last_week_dates]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_week, y=y_last, mode='lines+markers+text', name='지난 주',
        line=dict(width=2, dash='dot'),
        customdata=last_dates_str,
        hovertemplate="%{customdata}<br>지난 주: %{y:,}부<extra></extra>",
        text=[f"{v:,}" if v else "" for v in y_last],
        textposition='top center',
        textfont={'size': 11}
    ))
    fig.add_trace(go.Scatter(
        x=x_week, y=y_this, mode='lines+markers+text', name='이번 주',
        line=dict(width=3),
        customdata=this_dates_str,
        hovertemplate="%{customdata}<br>이번 주: %{y:,}부<extra></extra>",
        text=[f"{v:,}" if v else "" for v in y_this],
        textposition='top center',
        textfont={'size': 11}
    ))
    fig.update_layout(title=f'주간 발주량 비교 (기준일: {now.strftime("%Y-%m-%d")})',
                      xaxis_title='', yaxis_title='발주 부수', template='plotly_white', height=260,
                      margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation='h', x=1, xanchor='right', y=1.1))
    return fig


def figure_months_1to12(df_monthly: pd.DataFrame, start_year=2022, current_year=None) -> go.Figure:
    if current_year is None:
        current_year = datetime.now(KST).year
    d = df_monthly[(df_monthly['연도'] >= start_year) & (
        df_monthly['연도'] <= current_year)].copy()
    if d.empty:
        return go.Figure()
    pivot = d.pivot_table(index='월번호', columns='연도',
                          values='발주량', aggfunc='sum').sort_index()
    pivot = pivot[[c for c in pivot.columns if c <= current_year]]
    fig = px.line(pivot, x=pivot.index, y=pivot.columns, markers=True,
                  title=f'월별 발주량 (1~12월, {start_year}~{current_year})')
    fig.update_layout(xaxis_title='', yaxis_title='발주량', template='plotly_white', height=300,
                      margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation='h', x=1, xanchor='right', y=1.1))
    fig.update_xaxes(dtick=1)
    return fig


def yoy_line_value_bar_rate(d: pd.DataFrame, value_col: str, title: str, baseline_year: int) -> go.Figure:
    if d.empty:
        return go.Figure()
    d = d.copy()
    d = d[d['연도'] <= baseline_year]
    d = d.sort_values(['연도', '월번호'])
    if value_col not in d.columns:
        d[value_col] = 0
    d['prev_year'] = d.groupby('월번호')[value_col].shift(1)
    d['YoY%'] = ((d[value_col] - d['prev_year']) /
                 d['prev_year'].replace({0: pd.NA})) * 100

    fig = go.Figure()
    for y, sub in d.groupby('연도'):
        fig.add_trace(go.Scatter(
            x=sub['월번호'], y=sub[value_col], mode='lines+markers', name=f'{y}년 ({value_col})'
        ))
    base = d[d['연도'] == baseline_year]
    fig.add_trace(go.Bar(
        x=base['월번호'], y=base['YoY%'], name=f'{baseline_year} YoY%',
        yaxis='y2', opacity=0.6,
        hovertemplate="증감율: %{y:.1f}%%<extra></extra>"
    ))

    fig.update_layout(
        title=title, xaxis_title='', yaxis_title=value_col,
        yaxis2=dict(title='YoY %', overlaying='y',
                    side='right', showgrid=False),
        template='plotly_white', height=420,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation='h', x=1, xanchor='right', y=1.12)
    )
    fig.update_xaxes(dtick=1)
    return fig


def build_metric_figs(df_monthly: pd.DataFrame, baseline_year: int):
    d = df_monthly[(df_monthly['연도'] >= 2022) & (
        df_monthly['연도'] <= baseline_year)].copy()
    if '흑백출력량' not in d.columns:
        alt_bw = next((c for c in d.columns if '흑백' in c), None)
        if alt_bw:
            d = d.rename(columns={alt_bw: '흑백출력량'})
        else:
            d['흑백출력량'] = 0
    if '컬러출력량' not in d.columns:
        alt_c = next((c for c in d.columns if ('컬러' in c or '칼라' in c)), None)
        if alt_c:
            d = d.rename(columns={alt_c: '컬러출력량'})
        else:
            d['컬러출력량'] = 0

    return {
        'avg_per_day': yoy_line_value_bar_rate(d, '일평균발주량', '월별 일평균 발주량 + YoY%', baseline_year),
        'monthly_total': yoy_line_value_bar_rate(d, '발주량', '월 총 발주량 + YoY%', baseline_year),
        'bw_pages': yoy_line_value_bar_rate(d, '흑백출력량', '월별 흑백 페이지 + YoY%', baseline_year),
        'color_pages': yoy_line_value_bar_rate(d, '컬러출력량', '월별 컬러 페이지 + YoY%', baseline_year),
    }


def compute_kpis(df_daily: pd.DataFrame, year: int):
    if df_daily.empty:
        return 0, 0.0, 0
    year_df = df_daily[df_daily['연도'] == year]
    total_orders = int(year_df['총발주부수'].sum()) if not year_df.empty else 0
    days_count = int(year_df['date_only'].nunique()
                     ) if not year_df.empty else 0
    avg_per_day = round(total_orders / days_count, 2) if days_count else 0.0
    return total_orders, avg_per_day, days_count


def kpi_card(title, value, subtitle):
    return html.Div(style={
        'flex': '1 1 260px', 'background': 'white', 'borderRadius': '14px', 'padding': '14px 16px',
        'boxShadow': '0 6px 18px rgba(0,0,0,0.08)', 'minWidth': '240px'
    }, children=[
        html.Div(title, style={'fontSize': '0.95rem', 'color': '#666',
                 'marginBottom': '6px', 'fontWeight': '600'}),
        html.Div(value, style={'fontSize': '1.8rem', 'fontWeight': '800'}),
        html.Div(subtitle, style={'fontSize': '0.85rem',
                 'color': '#8a8a8a', 'marginTop': '4px'}),
    ])


def kpi_row(total_orders, avg_per_day, days_count, year):
    return html.Div(style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap', 'marginBottom': '8px'}, children=[
        kpi_card(f"{year}년 총 발주량", f"{total_orders:,}", "Total Orders"),
        kpi_card(f"{year}년 일 평균 발주량", f"{avg_per_day:,}", "Avg / Working Day"),
        kpi_card(f"{year}년 총 발주 일수", f"{days_count:,}일", "Working Days Count"),
    ])


def previous_years_table(df_daily: pd.DataFrame, current_year: int):
    if df_daily.empty:
        return html.Table([])
    years = sorted(df_daily['연도'].unique())
    rows = []
    for y in years:
        if y == current_year:
            continue
        tot, avg, days = compute_kpis(df_daily, y)
        rows.append(html.Tr([html.Td(f"{y}년"), html.Td(
            f"{tot:,}"), html.Td(f"{avg:,}"), html.Td(f"{days:,}")]))

    header = html.Tr([html.Th("연도"), html.Th("총 발주량"),
                     html.Th("일 평균 발주량"), html.Th("총 발주 일수")])
    return html.Table([header] + rows, style={'width': '100%', 'borderCollapse': 'collapse'}, className="kpi-table")


external_stylesheets = [{
    "href": "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css",
    "rel": "stylesheet"
}]
app = dash.Dash(__name__,
                title="발주량 분석 대시보드",
                meta_tags=[{"name": "viewport",
                            "content": "width=device-width, initial-scale=1"}],
                external_stylesheets=external_stylesheets)
server = app.server

try:
    _raw_daily, _raw_monthly = load_data_from_gsheet()
except Exception as e:
    print("[ERROR] 시트 로드 실패:", e)
    _raw_daily, _raw_monthly = pd.DataFrame(), pd.DataFrame()

DF_DAILY = cleanse_daily(_raw_daily)
DF_MONTHLY = cleanse_monthly(_raw_monthly)

CURRENT_YEAR = datetime.now(KST).year
years_options = [{'label': f'{int(y)}년', 'value': int(y)} for y in sorted(
    DF_DAILY['연도'].unique()) if int(y) <= CURRENT_YEAR] if not DF_DAILY.empty else []

fig_week = figure_weekly(DF_DAILY) if not DF_DAILY.empty else go.Figure()
fig_months = figure_months_1to12(
    DF_MONTHLY, start_year=2022, current_year=CURRENT_YEAR) if not DF_MONTHLY.empty else go.Figure()
metric_figs = build_metric_figs(DF_MONTHLY, baseline_year=CURRENT_YEAR) if not DF_MONTHLY.empty else {
    'avg_per_day': go.Figure(), 'monthly_total': go.Figure(), 'bw_pages': go.Figure(), 'color_pages': go.Figure()}

BASE_STYLE = {'fontFamily': 'Noto Sans KR, Malgun Gothic, Arial'}
CARD_STYLE = {'background': 'white', 'borderRadius': '12px',
              'padding': '14px', 'boxShadow': '0 4px 14px rgba(0,0,0,0.08)'}

app.layout = html.Div(style={'maxWidth': '1100px', 'margin': '0 auto', 'padding': '16px', **BASE_STYLE}, children=[
    html.H1("발주량 분석 대시보드", style={
            'textAlign': 'center', 'marginBottom': '6px'}),
    html.P("구글 시트 데이터 기반 · 2021~현재", style={
           'textAlign': 'center', 'marginBottom': '14px', 'color': '#666'}),

    html.Div(style={'display': 'flex', 'gap': '10px', 'justifyContent': 'center', 'alignItems': 'center', 'marginBottom': '8px'}, children=[
        html.Span("KPI 연도 선택:", style={'fontWeight': '600'}),
        dcc.Dropdown(id='year-select', options=years_options,
                     value=(years_options[-1]['value']
                            if years_options else CURRENT_YEAR),
                     clearable=False, style={'width': '220px'}),
        html.Div(id='kpi-refresh-status',
                 style={'marginLeft': '12px', 'color': '#888'})
    ]),

    html.Div(id='kpi-cards'),

    html.Details(open=False, children=[
        html.Summary("지난 연도 KPI 펼치기 / 접기"),
        html.Div(id='prev-years-kpi', style={'marginTop': '8px'})
    ], style={**CARD_STYLE, 'marginBottom': '16px'}),

    html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr', 'gap': '12px', 'marginBottom': '12px'}, children=[
        html.Div([html.H3("주간 발주량 비교 (오늘 기준 5영업일 + 지난주 같은 요일)", style={'marginBottom': '4px', 'fontSize': '1.05rem'}),
                  dcc.Graph(id='weekly-chart', figure=fig_week, style={'height': '280px'})], style=CARD_STYLE),
    ]),

    html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr', 'gap': '12px', 'marginBottom': '12px'}, children=[
        html.Div([html.H3("월별 발주량 (1~12월, 2022~현재)", style={'marginBottom': '4px', 'fontSize': '1.05rem'}),
                  dcc.Graph(id='months-1to12-chart', figure=fig_months, style={'height': '320px'})], style=CARD_STYLE),
    ]),

    html.Div(style={**CARD_STYLE, 'padding': '0px'}, children=[
        dcc.Tabs(id="metric-tabs", value="avg", children=[
            dcc.Tab(label="일평균 발주량", value="avg"),
            dcc.Tab(label="월 총 발주량", value="total"),
            dcc.Tab(label="흑백 페이지", value="bw"),
            dcc.Tab(label="컬러 페이지", value="color"),
        ]),
        html.Div(id="metric-tab-content", style={'padding': '8px 4px'})
    ]),

    html.Div(style={'textAlign': 'center', 'color': '#888',
             'marginTop': '16px', 'fontSize': '.9rem'}, children="© 2025 발주량 분석 대시보드")
])


@callback(
    Output('kpi-cards', 'children'),
    Output('prev-years-kpi', 'children'),
    Output('kpi-refresh-status', 'children'),
    Input('year-select', 'value')
)
def update_kpis(selected_year):
    try:
        tot, avg, days = compute_kpis(DF_DAILY, selected_year)
        cards = kpi_row(tot, avg, days, selected_year)
        prev_tbl = previous_years_table(DF_DAILY, selected_year)
        stamp = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")
        return cards, prev_tbl, f"업데이트: {stamp}"
    except Exception as e:
        return html.Div("KPI 계산 오류"), html.Div(f"오류: {e}"), ""


@callback(
    Output("metric-tab-content", "children"),
    Input("metric-tabs", "value")
)
def switch_metric_tab(tab_value):
    figs = {
        "avg": metric_figs['avg_per_day'],
        "total": metric_figs['monthly_total'],
        "bw": metric_figs['bw_pages'],
        "color": metric_figs['color_pages'],
    }
    fig = figs.get(tab_value, go.Figure())
    fig.update_layout(height=460)
    return dcc.Graph(figure=fig, style={'height': '480px'})


if __name__ == '__main__':
    host = os.getenv('DASH_HOST', '127.0.0.1')
    port = int(os.getenv('DASH_PORT', '8090'))
    print(f"브라우저에서 http://{host}:{port} 으로 접속하세요.")
    app.run(debug=True, host=host, port=port)

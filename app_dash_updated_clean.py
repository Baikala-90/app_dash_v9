
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

# ---------------------------------------------
# Helpers
# ---------------------------------------------


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
    print(f"[INFO] Open spreadsheet: {url}")
    return client.open_by_url(url)

# ---------------------------------------------
# Load
# ---------------------------------------------


def load_data_from_gsheet():
    sh = open_sheet()
    daily_name = os.getenv("DAILY_SHEET_NAME", "일별 발주량 외").strip()
    monthly_name = os.getenv("MONTHLY_SHEET_NAME", "월별 발주량").strip()

    # 일별
    ws_d = sh.worksheet(daily_name)
    vals_d = ws_d.get_all_values()
    di = int(os.getenv("DAILY_HEADER_INDEX", "0"))
    headers_d = [norm(h) for h in vals_d[di]]
    df_daily_raw = pd.DataFrame(vals_d[di+1:], columns=headers_d)

    # 월별
    ws_m = sh.worksheet(monthly_name)
    vals_m = ws_m.get_all_values()
    mi = int(os.getenv("MONTHLY_HEADER_INDEX", "1"))
    headers_m = [norm(h) for h in vals_m[mi]]
    df_monthly_raw = pd.DataFrame(vals_m[mi+1:], columns=headers_m)

    print(
        f"[LOAD] daily rows: {len(df_daily_raw)}, monthly rows: {len(df_monthly_raw)}")
    return df_daily_raw, df_monthly_raw

# ---------------------------------------------
# Cleanse - Daily
# ---------------------------------------------


def cleanse_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=['날짜', '총발주부수', '흑백페이지', '컬러페이지', 'date_only', '연도', '월'])

    d = df.copy()
    d = d.rename(columns={c: norm(c) for c in d.columns})

    # 베이스 컬럼 식별
    # 열 순서 가정: [날짜, 총발주종수(B), 총발주부수(C), 흑백페이지(D), 컬러페이지(E), ...]
    # 일부 시트에서는 컬럼명이 약간 다를 수 있으므로 유연하게 매핑
    col_date = next((c for c in d.columns if c in [
                    '날짜', '날짜(년월일)', '일자', '날']), d.columns[0])

    col_cnt = next((c for c in d.columns if c in [
                   '총발주종수', '총발주건수', '총발주건', '종수']), None)  # B
    col_total = next((c for c in d.columns if c in [
                     '총발주부수', '총발주량', '총발주부', '총발주수', '총발주']), None)  # C
    col_bw = next((c for c in d.columns if '흑백' in c), None)  # D
    col_color = next((c for c in d.columns if (
        '컬러' in c or '칼라' in c)), None)  # E

    # --- 1) 연도 헤더 처리: '2025년' 같은 행은 헤더로만 사용하고 데이터행 아님 ---
    # 연도 헤더를 추출해 내려 채움
    d['__year_header__'] = d[col_date].astype(str).str.extract(
        r'(^\s*(\d{4})\s*년\s*$)', expand=True)[1]
    d['__year_header__'] = d['__year_header__'].ffill()

    # --- 2) 날짜 문자열 정규화
    # 케이스 A: 'YYYY.MM.DD' → 그대로
    # 케이스 B: 'MM.DD (요일)' 또는 'M.DD' → 연도 헤더 + '.' + MM.DD 조합으로 변환
    date_str = d[col_date].astype(str).str.strip()
    # 괄호 요일 제거
    date_str = date_str.str.replace(r'\(.+?\)', '', regex=True).str.strip()

    # A: YYYY.MM.DD 패턴
    full_pat = re.compile(r'^\d{4}[./-]\d{1,2}[./-]\d{1,2}$')
    partial_pat = re.compile(r'^\d{1,2}[./-]\d{1,2}$')

    def make_full_date(row):
        s = row['__date_raw__']
        if full_pat.match(s):
            return s.replace('-', '.')
        if partial_pat.match(s):
            year = row['__year_header__']
            if pd.notna(year):
                return f"{year}.{s.replace('-', '.')}"
        return None

    d['__date_raw__'] = date_str
    d['__date_full__'] = d.apply(make_full_date, axis=1)
    # 최종 날짜 파싱
    d['날짜'] = pd.to_datetime(
        d['__date_full__'], errors='coerce', format='mixed', yearfirst=True)

    # 'YYYY년' 헤더 행이나 날짜 파싱 실패 행 제거
    d = d.dropna(subset=['날짜']).copy()

    # --- 3) B,C,D,E가 모두 공백이면 "기록하지 않은 날/비영업일"로 간주하여 제외 ---
    def is_blank(col):
        if col is None or col not in d.columns:
            return True  # 해당 컬럼 자체가 없으면 공백 취급
        return d[col].astype(str).str.strip().eq('').fillna(True)

    blanks = is_blank(col_cnt) & is_blank(
        col_total) & is_blank(col_bw) & is_blank(col_color)
    before = len(d)
    d = d.loc[~blanks].copy()  # 전부 공백인 행 제외
    print(f"[CLEAN DAILY] dropped non-record/holiday rows: {before - len(d)}")

    # --- 4) 숫자화 (쉼표 제거 후)
    for src, std in [(col_total, '총발주부수'), (col_bw, '흑백페이지'), (col_color, '컬러페이지')]:
        if src and src in d.columns:
            d[std] = pd.to_numeric(d[src].astype(str).str.replace(
                ',', '', regex=False), errors='coerce').fillna(0)
        else:
            d[std] = 0

    d['date_only'] = d['날짜'].dt.date
    d['연도'] = d['날짜'].dt.year
    d['월'] = d['날짜'].dt.month

    # 요약 로그
    if not d.empty:
        print("[CLEAN DAILY] 기간:", d['날짜'].min(),
              "~", d['날짜'].max(), "행:", len(d))
        print("[CLEAN DAILY] 샘플:", d[['날짜', '총발주부수', '흑백페이지', '컬러페이지']].head(
            3).to_dict('records'))
    else:
        print("[CLEAN DAILY] 결과가 비었습니다. (헤더 인덱스/컬럼명 확인 필요)")

    return d[['날짜', 'date_only', '연도', '월', '총발주부수', '흑백페이지', '컬러페이지']]

# ---------------------------------------------
# Cleanse - Monthly
# ---------------------------------------------


def cleanse_monthly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=['월DT', '발주량', '발주일수', '일평균발주량', '흑백출력량', '컬러출력량', '연도', '월번호'])

    d = df.copy()
    d = d.rename(columns={c: norm(c) for c in d.columns})
    month_col = '월' if '월' in d.columns else d.columns[0]

    # 표준화
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

    if not d.empty:
        print("[CLEAN MONTHLY] 기간:", d['월DT'].min(),
              "~", d['월DT'].max(), "행:", len(d))
        print("[CLEAN MONTHLY] 샘플:", d[['월DT', '발주량', '발주일수',
              '일평균발주량', '흑백출력량', '컬러출력량']].head(3).to_dict('records'))
    return d[['월DT', '연도', '월번호', '발주량', '발주일수', '일평균발주량', '흑백출력량', '컬러출력량']]

# ---------------------------------------------
# Figures (compact)
# ---------------------------------------------


def last_5_business_days_upto_today(now_kst: datetime):
    dates = []
    d = now_kst.date()
    while len(dates) < 5:
        if d.weekday() < 5:
            dates.append(d)
        d -= timedelta(days=1)
    return list(reversed(dates))


def figure_weekly(df_daily: pd.DataFrame) -> go.Figure:
    now = datetime.now(KST)
    this_week_dates = last_5_business_days_upto_today(now)
    last_week_dates = [d - timedelta(days=7) for d in this_week_dates]

    m = df_daily.set_index('date_only')[
        '총발주부수'].to_dict() if not df_daily.empty else {}
    y_this = [m.get(d, 0) for d in this_week_dates]
    y_last = [m.get(d, 0) for d in last_week_dates]
    x_labels = [pd.Timestamp(d).strftime('%m/%d(%a)') for d in this_week_dates]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_labels, y=y_last, mode='lines+markers',
                  name='지난 주', line=dict(width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=x_labels, y=y_this,
                  mode='lines+markers', name='이번 주', line=dict(width=3)))
    fig.update_layout(title=f'주간 발주량 비교 (기준일: {now.strftime("%Y-%m-%d")})',
                      xaxis_title='', yaxis_title='발주 부수', template='plotly_white', height=260,
                      margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation='h', x=1, xanchor='right', y=1.1))
    return fig


def figure_months_1to12(df_monthly: pd.DataFrame, start_year=2022) -> go.Figure:
    d = df_monthly[df_monthly['연도'] >= start_year].copy()
    if d.empty:
        return go.Figure()
    pivot = d.pivot_table(index='월번호', columns='연도',
                          values='발주량', aggfunc='sum').sort_index()
    fig = px.line(pivot, x=pivot.index, y=pivot.columns,
                  markers=True, title='월별 발주량 (1~12월, 2022~현재)')
    fig.update_layout(xaxis_title='', yaxis_title='발주량', template='plotly_white', height=300,
                      margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation='h', x=1, xanchor='right', y=1.1))
    fig.update_xaxes(dtick=1)
    return fig


def add_yoy(fig: go.Figure, d: pd.DataFrame, value_col: str, title: str) -> go.Figure:
    if d.empty:
        return fig
    d = d.copy().sort_values(['연도', '월번호'])
    if value_col not in d.columns:
        d[value_col] = 0
    d['prev_year'] = d.groupby('월번호')[value_col].shift(1)
    d['YoY%'] = ((d[value_col] - d['prev_year']) /
                 d['prev_year'].replace({0: pd.NA})) * 100

    for y, sub in d.groupby('연도'):
        fig.add_trace(
            go.Bar(x=sub['월번호'], y=sub[value_col], name=f'{y}년', opacity=0.85))

    latest_year = d['연도'].max()
    yoy_latest = d[d['연도'] == latest_year]
    fig.add_trace(go.Scatter(x=yoy_latest['월번호'], y=yoy_latest['YoY%'],
                             name=f'{latest_year} YoY%', mode='lines+markers', yaxis='y2'))

    fig.update_layout(
        title=title, xaxis_title='', yaxis_title=value_col,
        yaxis2=dict(title='YoY %', overlaying='y',
                    side='right', showgrid=False),
        barmode='group', template='plotly_white', height=300,
        margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation='h', x=1, xanchor='right', y=1.12)
    )
    fig.update_xaxes(dtick=1)
    return fig


def build_metric_figs(df_monthly: pd.DataFrame):
    d = df_monthly[df_monthly['연도'] >= 2022].copy()
    figs = {}
    figs['avg_per_day'] = add_yoy(
        go.Figure(), d, '일평균발주량', '월별 일평균 발주량 + YoY%')
    figs['monthly_total'] = add_yoy(go.Figure(), d, '발주량', '월 총 발주량 + YoY%')
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
    figs['bw_pages'] = add_yoy(go.Figure(), d, '흑백출력량', '월별 흑백 페이지 + YoY%')
    figs['color_pages'] = add_yoy(go.Figure(), d, '컬러출력량', '월별 컬러 페이지 + YoY%')
    return figs

# ---------------------------------------------
# KPI
# ---------------------------------------------


def compute_kpis(df_daily: pd.DataFrame, year: int):
    if df_daily.empty:
        return 0, 0.0, 0
    # KPI 계산 시 "기록된 영업일" 기준 (B,C,D,E 모두 공백이었던 날은 이미 제거됨)
    year_df = df_daily[df_daily['연도'] == year]
    total_orders = int(year_df['총발주부수'].sum()) if not year_df.empty else 0
    days_count = int(year_df['date_only'].nunique()
                     ) if not year_df.empty else 0
    avg_per_day = round(total_orders / days_count, 2) if days_count else 0.0
    return total_orders, avg_per_day, days_count


def kpi_cards(total_orders, avg_per_day, days_count, year):
    return html.Div(className="kpi-grid", children=[
        html.Div(className="kpi-card",
                 children=[html.H4(f"{year}년 총 발주량"), html.H2(f"{total_orders:,}")]),
        html.Div(className="kpi-card",
                 children=[html.H4(f"{year}년 일 평균 발주량"), html.H2(f"{avg_per_day:,}")]),
        html.Div(className="kpi-card",
                 children=[html.H4(f"{year}년 총 발주 일수"), html.H2(f"{days_count:,}일")]),
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


# ---------------------------------------------
# App
# ---------------------------------------------
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

# 데이터 로드
try:
    _raw_daily, _raw_monthly = load_data_from_gsheet()
except Exception as e:
    print("[ERROR] 시트 로드 실패:", e)
    _raw_daily, _raw_monthly = pd.DataFrame(), pd.DataFrame()

DF_DAILY = cleanse_daily(_raw_daily)
DF_MONTHLY = cleanse_monthly(_raw_monthly)

CURRENT_YEAR = datetime.now(KST).year
years_options = [{'label': f'{int(y)}년', 'value': int(y)} for y in sorted(
    DF_DAILY['연도'].unique())] if not DF_DAILY.empty else []

fig_week = figure_weekly(DF_DAILY) if not DF_DAILY.empty else go.Figure()
fig_months = figure_months_1to12(
    DF_MONTHLY, start_year=2022) if not DF_MONTHLY.empty else go.Figure()
metric_figs = build_metric_figs(DF_MONTHLY) if not DF_MONTHLY.empty else {'avg_per_day': go.Figure(
), 'monthly_total': go.Figure(), 'bw_pages': go.Figure(), 'color_pages': go.Figure()}

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

    html.Div(id='kpi-cards', style={'display': 'grid', 'gridTemplateColumns':
             'repeat(auto-fit, minmax(200px, 1fr))', 'gap': '10px', 'marginBottom': '8px'}),

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

    html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(320px, 1fr))', 'gap': '12px'}, children=[
        html.Div([html.H3("월별 일평균 발주량 + YoY%", style={'marginBottom': '4px', 'fontSize': '1.05rem'}),
                  dcc.Graph(id='fig-avg', figure=metric_figs['avg_per_day'], style={'height': '320px'})], style=CARD_STYLE),
        html.Div([html.H3("월 총 발주량 + YoY%", style={'marginBottom': '4px', 'fontSize': '1.05rem'}),
                  dcc.Graph(id='fig-total', figure=metric_figs['monthly_total'], style={'height': '320px'})], style=CARD_STYLE),
        html.Div([html.H3("월별 흑백 페이지 + YoY%", style={'marginBottom': '4px', 'fontSize': '1.05rem'}),
                  dcc.Graph(id='fig-bw', figure=metric_figs['bw_pages'], style={'height': '320px'})], style=CARD_STYLE),
        html.Div([html.H3("월별 컬러 페이지 + YoY%", style={'marginBottom': '4px', 'fontSize': '1.05rem'}),
                  dcc.Graph(id='fig-color', figure=metric_figs['color_pages'], style={'height': '320px'})], style=CARD_STYLE),
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
        cards = kpi_cards(tot, avg, days, selected_year)
        prev_tbl = previous_years_table(DF_DAILY, selected_year)
        stamp = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")
        return cards, prev_tbl, f"업데이트: {stamp}"
    except Exception as e:
        return html.Div("KPI 계산 오류"), html.Div(f"오류: {e}"), ""


if __name__ == '__main__':
    host = os.getenv('DASH_HOST', '127.0.0.1')
    port = int(os.getenv('DASH_PORT', '8090'))
    print("발주량 분석 대시보드를 시작합니다...")
    print(f"브라우저에서 http://{host}:{port} 으로 접속하세요.")
    app.run(debug=True, host=host, port=port)

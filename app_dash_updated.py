
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
import os
from dotenv import load_dotenv
import pytz
import re

load_dotenv()
warnings.filterwarnings("ignore", message="Could not infer format, so each element will be parsed individually.*")
KST = pytz.timezone('Asia/Seoul')

# -------------------- utils --------------------
def normalize_header(h):
    # remove spaces & newlines
    return re.sub(r'\s+', '', (h or '')).strip()

def guess_daily_header_row(values):
    for i, row in enumerate(values[:20]):
        cell = (row[0] or "").strip()
        # looks like '2021.02.19 (금)' or '2021-02-19'
        if re.search(r'\d{4}[./-]\d{1,2}[./-]\d{1,2}', cell):
            # header likely one row above; but in sample, data starts at that row,
            # so header is i-1 or first line? We'll use i-1 if it contains '날'
            if i>0 and ('날' in ''.join(values[i-1])):
                return i-1
            return i-1 if i>0 else 0
    return 3

def guess_monthly_header_row(values):
    for i, row in enumerate(values[:20]):
        cell = (row[0] or "").replace(" ", "")
        # '2021년3월' data row typically, header one row above where first cell == '월'
        if re.match(r'^\d{4}년\d{1,2}월$', cell):
            # find nearest previous row that contains '월' as first cell or includes it
            for j in range(i-1, max(-1, i-5), -1):
                if j>=0 and (values[j] and '월' in (values[j][0] or '')):
                    return j
            return i-1 if i>0 else 0
    # fallback: find row whose first cell == '월'
    for i, row in enumerate(values[:10]):
        if (row and (row[0] or '').strip() == '월'):
            return i
    return 2

def open_sheet():
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds_file = 'credentials.json'
    if not os.path.exists(creds_file):
        raise FileNotFoundError("credentials.json 파일을 찾을 수 없습니다.")
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_file, scope)
    client = gspread.authorize(creds)
    sheet_url = os.getenv('SPREADSHEET_URL')
    if not sheet_url:
        raise EnvironmentError(".env의 SPREADSHEET_URL이 설정되어 있지 않습니다.")
    return client.open_by_url(sheet_url)

def load_data_from_gsheet():
    sh = open_sheet()

    daily_name = os.getenv('DAILY_SHEET_NAME', '일별 발주량 외')
    monthly_name = os.getenv('MONTHLY_SHEET_NAME', '월별 발주량')

    daily_ws = sh.worksheet(daily_name)
    daily_vals = daily_ws.get_all_values()
    di = guess_daily_header_row(daily_vals)
    daily_headers = [normalize_header(h) for h in daily_vals[di]]
    daily_rows = daily_vals[di+1:]
    df_daily = pd.DataFrame(daily_rows, columns=daily_headers)

    monthly_ws = sh.worksheet(monthly_name)
    monthly_vals = monthly_ws.get_all_values()
    mi = guess_monthly_header_row(monthly_vals)
    monthly_headers = [normalize_header(h) for h in monthly_vals[mi]]
    monthly_rows = monthly_vals[mi+1:]
    df_monthly = pd.DataFrame(monthly_rows, columns=monthly_headers)

    return df_daily, df_monthly

def cleanse_daily(df_daily: pd.DataFrame) -> pd.DataFrame:
    d = df_daily.copy()
    # find columns by normalized names
    col_date = next((c for c in d.columns if c in ['날짜','날짜(년월일)','일자','날']), None)
    if not col_date:
        # try columns that include '날'
        col_date = next((c for c in d.columns if '날' in c or '일자' in c), None)
    col_total = next((c for c in d.columns if c in ['총발주부수','총발주량','총발주부','총발주수','총발주']), None)
    if not col_total:
        col_total = next((c for c in d.columns if '총발주' in c and ('부수' in c or '량' in c)), None)
    col_bw = next((c for c in d.columns if c in ['흑백페이지','흑백출력량','흑백페이지수']), None)
    if not col_bw:
        col_bw = next((c for c in d.columns if '흑백' in c), None)
    col_color = next((c for c in d.columns if c in ['컬러페이지','컬러출력량','컬러페이지수','칼라페이지']), None)
    if not col_color:
        col_color = next((c for c in d.columns if '컬러' in c or '칼라' in c), None)

    # rename standardized
    rename_map = {}
    if col_date: rename_map[col_date] = '날짜'
    if col_total: rename_map[col_total] = '총발주부수'
    if col_bw: rename_map[col_bw] = '흑백페이지'
    if col_color: rename_map[col_color] = '컬러페이지'
    d = d.rename(columns=rename_map)

    # drop empty rows
    if '날짜' in d.columns:
        d = d[d['날짜'].astype(str).str.strip() != '']

    # parse date
    if '날짜' in d.columns:
        # remove day-of-week parentheses (e.g., '2021.02.19 (금)')
        d['날짜'] = d['날짜'].astype(str).str.replace(r'\(.+\)', '', regex=True).str.strip()
        d['날짜'] = pd.to_datetime(d['날짜'], errors='coerce', format='mixed', yearfirst=True)
        d.dropna(subset=['날짜'], inplace=True)

    # numeric
    for c in ['총발주부수','흑백페이지','컬러페이지']:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c].astype(str).str.replace(',', '', regex=False), errors='coerce').fillna(0)

    d['date_only'] = d['날짜'].dt.date
    d['연도'] = d['날짜'].dt.year
    d['월'] = d['날짜'].dt.month
    # if total col didn't exist, create 0
    if '총발주부수' not in d.columns:
        d['총발주부수'] = 0
    if '흑백페이지' not in d.columns:
        d['흑백페이지'] = 0
    if '컬러페이지' not in d.columns:
        d['컬러페이지'] = 0
    return d

def cleanse_monthly(df_monthly: pd.DataFrame) -> pd.DataFrame:
    d = df_monthly.copy()
    # normalize names we care about
    # key columns after normalize: '월','발주량','발주일수','일평균발주량','흑백출력량','컬러출력량'
    col_month = next((c for c in d.columns if normalize_header(c) == '월'), None)
    if not col_month:
        # might already be normalized; take the first column as month text
        col_month = d.columns[0]
    # unify names
    rename_pairs = {}
    name_map = { '발주량':'발주량', '발주일수':'발주일수', '일평균발주량':'일평균발주량',
                 '흑백출력량':'흑백출력량', '컬러출력량':'컬러출력량' }
    for c in d.columns:
        n = normalize_header(c)
        if n in name_map:
            rename_pairs[c] = name_map[n]
    d = d.rename(columns=rename_pairs)

    # month parse
    month_series = d[col_month].astype(str).str.replace(' ', '')
    d['월DT'] = pd.to_datetime(month_series, format='%Y년%m월', errors='coerce')
    # some rows may have titles; dropna
    d = d.dropna(subset=['월DT'])

    # numeric clean
    for c in ['발주량','발주일수','일평균발주량','흑백출력량','컬러출력량']:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c].astype(str).str.replace(',', '', regex=False), errors='coerce').fillna(0)

    d['연도'] = d['월DT'].dt.year
    d['월번호'] = d['월DT'].dt.month
    return d

def last_5_business_days_upto_today(now_kst: datetime):
    dates = []
    d = now_kst.date()
    while len(dates) < 5:
        if d.weekday() < 5:
            dates.append(d)
        d -= timedelta(days=1)
    return list(reversed(dates))

# -------------------- figures --------------------
def figure_weekly(df_daily: pd.DataFrame) -> go.Figure:
    now = datetime.now(KST)
    this_week_dates = last_5_business_days_upto_today(now)
    last_week_dates = [d - timedelta(days=7) for d in this_week_dates]

    m = df_daily.set_index('date_only')['총발주부수'].to_dict()
    y_this = [m.get(d, 0) for d in this_week_dates]
    y_last = [m.get(d, 0) for d in last_week_dates]
    x_labels = [pd.Timestamp(d).strftime('%m/%d(%a)') for d in this_week_dates]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_labels, y=y_last, mode='lines+markers', name='지난 주', line=dict(width=3, dash='dash')))
    fig.add_trace(go.Scatter(x=x_labels, y=y_this, mode='lines+markers', name='이번 주', line=dict(width=3)))
    fig.update_layout(title=f'주간 발주량 비교 (기준일: {now.strftime("%Y-%m-%d")})',
                      xaxis_title='요일', yaxis_title='총 발주 부수', template='plotly_white', height=400)
    return fig

def figure_months_1to12(df_monthly: pd.DataFrame, start_year=2022) -> go.Figure:
    d = df_monthly[df_monthly['연도'] >= start_year].copy()
    if d.empty:
        return go.Figure()
    pivot = d.pivot_table(index='월번호', columns='연도', values='발주량', aggfunc='sum').sort_index()
    fig = px.line(pivot, x=pivot.index, y=pivot.columns, markers=True, title='월별 발주량 (1~12월, 2022~현재)')
    fig.update_layout(xaxis_title='월', yaxis_title='발주량', template='plotly_white', height=420)
    fig.update_xaxes(dtick=1)
    return fig

def add_yoy(fig: go.Figure, d: pd.DataFrame, value_col: str, title: str) -> go.Figure:
    if value_col not in d.columns:  # guard
        d[value_col] = 0
    d = d.copy().sort_values(['연도','월번호'])
    d['prev_year'] = d.groupby('월번호')[value_col].shift(1)
    d['YoY%'] = ((d[value_col] - d['prev_year']) / d['prev_year'].replace({0: pd.NA})) * 100

    for y, sub in d.groupby('연도'):
        fig.add_trace(go.Bar(x=sub['월번호'], y=sub[value_col], name=f'{y}년 {value_col}', opacity=0.85))

    latest_year = d['연도'].max()
    yoy_latest = d[d['연도'] == latest_year]
    fig.add_trace(go.Scatter(x=yoy_latest['월번호'], y=yoy_latest['YoY%'],
                             name=f'{latest_year}년 전년 대비 증감율(%)', mode='lines+markers', yaxis='y2'))

    fig.update_layout(
        title=title, xaxis_title='월', yaxis_title=value_col,
        yaxis2=dict(title='YoY %', overlaying='y', side='right', showgrid=False),
        barmode='group', template='plotly_white', height=420,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    fig.update_xaxes(dtick=1)
    return fig

def build_metric_figs(df_monthly: pd.DataFrame):
    d = df_monthly[df_monthly['연도'] >= 2022].copy()
    figs = {}
    figs['avg_per_day'] = add_yoy(go.Figure(), d, '일평균발주량', '월별 일평균 발주량 + YoY% (2022~현재)')
    figs['monthly_total'] = add_yoy(go.Figure(), d, '발주량', '월 총 발주량 + YoY% (2022~현재)')
    # attempt find columns for bw/color
    if '흑백출력량' not in d.columns:
        # sometimes monthly sheet may have '흑백페이지' aggregated monthly; try to find any
        alt_bw = next((c for c in d.columns if '흑백' in c), None)
        if alt_bw:
            d = d.rename(columns={alt_bw: '흑백출력량'})
        else:
            d['흑백출력량'] = 0
    if '컬러출력량' not in d.columns:
        alt_c = next((c for c in d.columns if '컬러' in c or '칼라' in c), None)
        if alt_c:
            d = d.rename(columns={alt_c: '컬러출력량'})
        else:
            d['컬러출력량'] = 0
    figs['bw_pages'] = add_yoy(go.Figure(), d, '흑백출력량', '월별 흑백 페이지 + YoY% (2022~현재)')
    figs['color_pages'] = add_yoy(go.Figure(), d, '컬러출력량', '월별 컬러 페이지 + YoY% (2022~현재)')
    return figs

# -------------------- kpis --------------------
def compute_kpis(df_daily: pd.DataFrame, year: int):
    year_df = df_daily[df_daily['연도'] == year]
    total_orders = int(year_df['총발주부수'].sum()) if not year_df.empty else 0
    days_count = int(year_df['date_only'].nunique()) if not year_df.empty else 0
    avg_per_day = round(total_orders / days_count, 2) if days_count else 0.0
    return total_orders, avg_per_day, days_count

def kpi_cards(total_orders, avg_per_day, days_count, year):
    return html.Div(className="kpi-grid", children=[
        html.Div(className="kpi-card", children=[html.H4(f"{year}년 총 발주량"), html.H2(f"{total_orders:,}")]),
        html.Div(className="kpi-card", children=[html.H4(f"{year}년 일 평균 발주량"), html.H2(f"{avg_per_day:,}")]),
        html.Div(className="kpi-card", children=[html.H4(f"{year}년 총 발주 일수"), html.H2(f"{days_count:,}일")]),
    ])

def previous_years_table(df_daily: pd.DataFrame, current_year: int):
    years = sorted(df_daily['연도'].unique())
    rows = []
    for y in years:
        if y == current_year:
            continue
        tot, avg, days = compute_kpis(df_daily, y)
        rows.append(html.Tr([html.Td(f"{y}년"), html.Td(f"{tot:,}"), html.Td(f"{avg:,}"), html.Td(f"{days:,}")]))

    header = html.Tr([html.Th("연도"), html.Th("총 발주량"), html.Th("일 평균 발주량"), html.Th("총 발주 일수")])
    return html.Table([header] + rows, style={'width':'100%','borderCollapse':'collapse'}, className="kpi-table")

# -------------------- Dash app --------------------
external_stylesheets = [{
    "href": "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css",
    "rel": "stylesheet"
}]
app = dash.Dash(__name__, 
                title="발주량 분석 대시보드",
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
                external_stylesheets=external_stylesheets)
server = app.server

# Load data
try:
    _df_daily_raw, _df_monthly_raw = load_data_from_gsheet()
except Exception as e:
    print("[에러] 시트 로드 실패:", e)
    _df_daily_raw, _df_monthly_raw = pd.DataFrame(), pd.DataFrame()

DF_DAILY = cleanse_daily(_df_daily_raw) if not _df_daily_raw.empty else pd.DataFrame(columns=['날짜','총발주부수','흑백페이지','컬러페이지','date_only','연도','월'])
DF_MONTHLY = cleanse_monthly(_df_monthly_raw) if not _df_monthly_raw.empty else pd.DataFrame(columns=['월DT','발주량','발주일수','일평균발주량','흑백출력량','컬러출력량','연도','월번호'])

CURRENT_YEAR = datetime.now(KST).year

fig_week = figure_weekly(DF_DAILY) if not DF_DAILY.empty else go.Figure()
fig_months = figure_months_1to12(DF_MONTHLY, start_year=2022) if not DF_MONTHLY.empty else go.Figure()
metric_figs = build_metric_figs(DF_MONTHLY) if not DF_MONTHLY.empty else {'avg_per_day':go.Figure(),'monthly_total':go.Figure(),'bw_pages':go.Figure(),'color_pages':go.Figure()}

BASE_STYLE = {'fontFamily': 'Noto Sans KR, Malgun Gothic, Arial'}
CARD_STYLE = {'background':'white','borderRadius':'12px','padding':'16px','boxShadow':'0 4px 16px rgba(0,0,0,0.1)'}

app.layout = html.Div(style={'maxWidth':'1400px','margin':'0 auto','padding':'20px', **BASE_STYLE}, children=[
    html.H1("발주량 분석 대시보드", style={'textAlign':'center','marginBottom':'10px'}),
    html.P("구글 시트 데이터 기반 · 2021~현재", style={'textAlign':'center','marginBottom':'20px'}),

    html.Div(style={'display':'flex','gap':'10px','justifyContent':'center','alignItems':'center','marginBottom':'10px'}, children=[
        html.Span("KPI 연도 선택:", style={'fontWeight':'600'}),
        dcc.Dropdown(
            id='year-select',
            options=[{'label': f'{y}년', 'value':int(y)} for y in sorted(DF_DAILY['연도'].unique())] if not DF_DAILY.empty else [],
            value=(sorted(DF_DAILY['연도'].unique())[-1] if not DF_DAILY.empty else datetime.now(KST).year),
            clearable=False, style={'width':'200px'}
        ),
        html.Div(id='kpi-refresh-status', style={'marginLeft':'12px', 'color':'#888'})
    ]),

    html.Div(id='kpi-cards', style={'display':'grid','gridTemplateColumns':'repeat(auto-fit, minmax(220px, 1fr))','gap':'12px','marginBottom':'10px'}),

    html.Details(open=False, children=[
        html.Summary("지난 연도 KPI 펼치기 / 접기"),
        html.Div(id='prev-years-kpi', style={'marginTop':'10px'})
    ], style={**CARD_STYLE,'marginBottom':'24px'}),

    html.Div(style={'display':'grid','gridTemplateColumns':'1fr','gap':'16px','marginBottom':'20px'}, children=[
        html.Div([html.H3("주간 발주량 비교 (오늘 기준 5영업일 + 지난주 같은 요일)", style={'marginBottom':'6px'}),
                  dcc.Graph(id='weekly-chart', figure=fig_week)], style=CARD_STYLE),
    ]),

    html.Div(style={'display':'grid','gridTemplateColumns':'1fr','gap':'16px','marginBottom':'20px'}, children=[
        html.Div([html.H3("월별 발주량 (1~12월, 2022~현재)", style={'marginBottom':'6px'}),
                  dcc.Graph(id='months-1to12-chart', figure=fig_months)], style=CARD_STYLE),
    ]),

    html.Div(style={'display':'grid','gridTemplateColumns':'1fr','gap':'16px'}, children=[
        html.Div([html.H3("월별 일평균 발주량 + YoY%", style={'marginBottom':'6px'}),
                  dcc.Graph(id='fig-avg', figure=metric_figs['avg_per_day'])], style=CARD_STYLE),
        html.Div([html.H3("월 총 발주량 + YoY%", style={'marginBottom':'6px'}),
                  dcc.Graph(id='fig-total', figure=metric_figs['monthly_total'])], style=CARD_STYLE),
        html.Div([html.H3("월별 흑백 페이지 + YoY%", style={'marginBottom':'6px'}),
                  dcc.Graph(id='fig-bw', figure=metric_figs['bw_pages'])], style=CARD_STYLE),
        html.Div([html.H3("월별 컬러 페이지 + YoY%", style={'marginBottom':'6px'}),
                  dcc.Graph(id='fig-color', figure=metric_figs['color_pages'])], style=CARD_STYLE),
    ]),

    html.Div(style={'textAlign':'center','color':'#888','marginTop':'24px'}, children="© 2025 발주량 분석 대시보드")
])

@callback(
    Output('kpi-cards','children'),
    Output('prev-years-kpi','children'),
    Output('kpi-refresh-status','children'),
    Input('year-select','value')
)
def update_kpis(selected_year):
    try:
        tot, avg, days = compute_kpis(DF_DAILY, selected_year)
        cards = kpi_cards(tot, avg, days, selected_year)
        prev_tbl = previous_years_table(DF_DAILY, selected_year)
        stamp = datetime.now(KST).strftime("%Y-%m-%d %H:%M")
        return cards, prev_tbl, f"업데이트: {stamp} KST"
    except Exception as e:
        return html.Div("KPI 계산 오류"), html.Div(f"오류: {e}"), ""

if __name__ == '__main__':
    print("발주량 분석 대시보드를 시작합니다...")
    print("브라우저에서 http://127.0.0.1:8088 으로 접속하세요.")
    app.run(debug=True, host=os.getenv('DASH_HOST','127.0.0.1'), port=int(os.getenv('DASH_PORT','8088')))

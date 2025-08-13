
import os
import re
from datetime import datetime, timedelta
import pytz
import warnings

from dotenv import load_dotenv
from flask import Flask, render_template, request

import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore", message="Could not infer format, so each element will be parsed individually.*")
load_dotenv()
KST = pytz.timezone('Asia/Seoul')

def normalize_header(h):
    return re.sub(r'\s+', '', (h or '')).strip()

def guess_daily_header_row(values):
    for i, row in enumerate(values[:20]):
        cell = (row[0] or "").strip()
        if re.search(r'\d{4}[./-]\d{1,2}[./-]\d{1,2}', cell):
            if i>0 and ('날' in ''.join(values[i-1])):
                return i-1
            return i-1 if i>0 else 0
    return 3

def guess_monthly_header_row(values):
    for i, row in enumerate(values[:20]):
        cell = (row[0] or "").replace(" ", "")
        if re.match(r'^\d{4}년\d{1,2}월$', cell):
            for j in range(i-1, max(-1, i-5), -1):
                if j>=0 and (values[j] and '월' in (values[j][0] or '')):
                    return j
            return i-1 if i>0 else 0
    for i, row in enumerate(values[:10]):
        if (row and (row[0] or '').strip() == '월'):
            return i
    return 2

def open_sheet():
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds_file = 'credentials.json'
    if not os.path.exists(creds_file):
        raise FileNotFoundError("credentials.json not found")
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_file, scope)
    client = gspread.authorize(creds)
    sheet_url = os.getenv('SPREADSHEET_URL')
    if not sheet_url:
        raise EnvironmentError("SPREADSHEET_URL missing in .env")
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
    col_date = next((c for c in d.columns if c in ['날짜','날짜(년월일)','일자','날']), None)
    if not col_date:
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

    rename_map = {}
    if col_date: rename_map[col_date] = '날짜'
    if col_total: rename_map[col_total] = '총발주부수'
    if col_bw: rename_map[col_bw] = '흑백페이지'
    if col_color: rename_map[col_color] = '컬러페이지'
    d = d.rename(columns=rename_map)

    if '날짜' in d.columns:
        d = d[d['날짜'].astype(str).str.strip() != '']
        d['날짜'] = d['날짜'].astype(str).str.replace(r'\(.+\)', '', regex=True).str.strip()
        d['날짜'] = pd.to_datetime(d['날짜'], errors='coerce', format='mixed', yearfirst=True)
        d.dropna(subset=['날짜'], inplace=True)

    for c in ['총발주부수','흑백페이지','컬러페이지']:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c].astype(str).str.replace(',', '', regex=False), errors='coerce').fillna(0)

    d['date_only'] = d['날짜'].dt.date
    d['연도'] = d['날짜'].dt.year
    d['월'] = d['날짜'].dt.month
    if '총발주부수' not in d.columns: d['총발주부수'] = 0
    if '흑백페이지' not in d.columns: d['흑백페이지'] = 0
    if '컬러페이지' not in d.columns: d['컬러페이지'] = 0
    return d

def cleanse_monthly(df_monthly: pd.DataFrame) -> pd.DataFrame:
    d = df_monthly.copy()
    col_month = next((c for c in d.columns if normalize_header(c) == '월'), None)
    if not col_month: col_month = d.columns[0]
    rename_pairs = {}
    for c in d.columns:
        n = normalize_header(c)
        if n == '발주량': rename_pairs[c] = '발주량'
        if n == '발주일수': rename_pairs[c] = '발주일수'
        if n == '일평균발주량': rename_pairs[c] = '일평균발주량'
        if n == '흑백출력량': rename_pairs[c] = '흑백출력량'
        if n == '컬러출력량': rename_pairs[c] = '컬러출력량'
    d = d.rename(columns=rename_pairs)

    month_series = d[col_month].astype(str).str.replace(' ', '')
    d['월DT'] = pd.to_datetime(month_series, format='%Y년%m월', errors='coerce')
    d = d.dropna(subset=['월DT'])

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

def figure_weekly(df_daily: pd.DataFrame):
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
                      xaxis_title='요일', yaxis_title='총 발주 부수', template='plotly_white', height=380)
    return fig

def figure_months_1to12(df_monthly: pd.DataFrame, start_year=2022):
    d = df_monthly[df_monthly['연도'] >= start_year].copy()
    if d.empty: return go.Figure()
    pivot = d.pivot_table(index='월번호', columns='연도', values='발주량', aggfunc='sum').sort_index()
    fig = px.line(pivot, x=pivot.index, y=pivot.columns, markers=True, title='월별 발주량 (1~12월, 2022~현재)')
    fig.update_layout(xaxis_title='월', yaxis_title='발주량', template='plotly_white', height=400)
    fig.update_xaxes(dtick=1)
    return fig

def add_yoy(fig, d: pd.DataFrame, value_col: str, title: str):
    if value_col not in d.columns: d[value_col] = 0
    d = d.copy().sort_values(['연도','월번호'])
    d['prev_year'] = d.groupby('월번호')[value_col].shift(1)
    d['YoY%'] = ((d[value_col] - d['prev_year']) / d['prev_year'].replace({0: pd.NA})) * 100

    for y, sub in d.groupby('연도'):
        fig.add_trace(go.Bar(x=sub['월번호'], y=sub[value_col], name=f'{y}년 {value_col}', opacity=0.85))

    latest_year = d['연도'].max()
    yoy_latest = d[d['연도'] == latest_year]
    fig.add_trace(go.Scatter(x=yoy_latest['월번호'], y=yoy_latest['YoY%'], name=f'{latest_year}년 전년 대비 증감율(%)', mode='lines+markers', yaxis='y2'))
    fig.update_layout(title=title, xaxis_title='월', yaxis_title=value_col,
                      yaxis2=dict(title='YoY %', overlaying='y', side='right', showgrid=False),
                      barmode='group', template='plotly_white', height=420,
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.update_xaxes(dtick=1)
    return fig

def figures_monthly_metrics(df_monthly: pd.DataFrame):
    d = df_monthly[df_monthly['연도'] >= 2022].copy()
    figs = {}
    figs['avg'] = add_yoy(go.Figure(), d, '일평균발주량', '월별 일평균 발주량 + YoY% (2022~현재)')
    figs['total'] = add_yoy(go.Figure(), d, '발주량', '월 총 발주량 + YoY% (2022~현재)')
    if '흑백출력량' not in d.columns:
        alt = next((c for c in d.columns if '흑백' in c), None)
        if alt: d = d.rename(columns={alt:'흑백출력량'})
        else: d['흑백출력량'] = 0
    if '컬러출력량' not in d.columns:
        alt = next((c for c in d.columns if '컬러' in c or '칼라' in c), None)
        if alt: d = d.rename(columns={alt:'컬러출력량'})
        else: d['컬러출력량'] = 0
    figs['bw'] = add_yoy(go.Figure(), d, '흑백출력량', '월별 흑백 페이지 + YoY% (2022~현재)')
    figs['color'] = add_yoy(go.Figure(), d, '컬러출력량', '월별 컬러 페이지 + YoY% (2022~현재)')
    return figs

app = Flask(__name__, template_folder='templates')

try:
    DF_DAILY_RAW, DF_MONTHLY_RAW = load_data_from_gsheet()
except Exception as e:
    print("[에러] 시트 로드 실패:", e)
    DF_DAILY_RAW, DF_MONTHLY_RAW = pd.DataFrame(), pd.DataFrame()

def safe_cleanse(df_raw, fn, default_cols):
    return fn(df_raw) if not df_raw.empty else pd.DataFrame(columns=default_cols)

DF_DAILY = safe_cleanse(DF_DAILY_RAW, cleanse_daily, ['날짜','총발주부수','흑백페이지','컬러페이지','date_only','연도','월'])
DF_MONTHLY = safe_cleanse(DF_MONTHLY_RAW, cleanse_monthly, ['월DT','발주량','발주일수','일평균발주량','흑백출력량','컬러출력량','연도','월번호'])

CURRENT_YEAR = datetime.now(KST).year

from flask import Markup

@app.route('/')
def home():
    try:
        selected_year = int(request.args.get('year', CURRENT_YEAR))
    except:
        selected_year = CURRENT_YEAR

    fig_week = figure_weekly(DF_DAILY) if not DF_DAILY.empty else go.Figure()
    fig_months = figure_months_1to12(DF_MONTHLY, start_year=2022) if not DF_MONTHLY.empty else go.Figure()
    metrics = figures_monthly_metrics(DF_MONTHLY) if not DF_MONTHLY.empty else {'avg':go.Figure(),'total':go.Figure(),'bw':go.Figure(),'color':go.Figure()}

    total = int(DF_DAILY[DF_DAILY['연도']==selected_year]['총발주부수'].sum()) if not DF_DAILY.empty else 0
    days = int(DF_DAILY[DF_DAILY['연도']==selected_year]['date_only'].nunique()) if not DF_DAILY.empty else 0
    avg = round(total / days, 2) if days else 0.0

    prev_rows = []
    if not DF_DAILY.empty:
        for y in sorted(DF_DAILY['연도'].unique()):
            if y == selected_year: continue
            tot = int(DF_DAILY[DF_DAILY['연도']==y]['총발주부수'].sum())
            dcount = int(DF_DAILY[DF_DAILY['연도']==y]['date_only'].nunique())
            av = round(tot / dcount, 2) if dcount else 0.0
            prev_rows.append({'year': y, 'total': f"{tot:,}", 'avg': f"{av:,}", 'days': f"{dcount:,}"})

    years_options = sorted(DF_DAILY['연도'].unique()) if not DF_DAILY.empty else [selected_year]

    return render_template('dashboard.html',
        years_options=years_options,
        selected_year=selected_year,
        kpi_total=f"{total:,}",
        kpi_avg=f"{avg:,}",
        kpi_days=f"{days:,}",
        prev_rows=prev_rows,
        fig_week_json=fig_week.to_json(),
        fig_months_json=fig_months.to_json(),
        fig_avg_json=metrics['avg'].to_json(),
        fig_total_json=metrics['total'].to_json(),
        fig_bw_json=metrics['bw'].to_json(),
        fig_color_json=metrics['color'].to_json()
    )

if __name__ == '__main__':
    host = os.getenv('FLASK_HOST', '127.0.0.1')
    port = int(os.getenv('FLASK_PORT', '8080'))
    debug = os.getenv('FLASK_DEBUG', '1') == '1'
    print("Flask dashboard running at http://%s:%s" % (host, port))
    app.run(host=host, port=port, debug=debug)

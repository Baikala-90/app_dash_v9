import os
import re
import calendar
from datetime import datetime, timedelta, date
import numpy as np

import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials

import dash
from dash import dcc, html, Input, Output, State, callback, clientside_callback
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go

from dotenv import load_dotenv
import pytz

# 새로 추가된 페이지 import
import billing_page

load_dotenv()
KST = pytz.timezone('Asia/Seoul')

# ===== 설정 =====
AUTO_REFRESH_MIN = int(os.getenv("AUTO_REFRESH_MIN", "15"))
HIGHLIGHT_THRESHOLD = float(
    os.getenv("HIGHLIGHT_THRESHOLD", "0.30"))  # 30% 변동 시 하이라이트
SPREADSHEET_URL = os.getenv("SPREADSHEET_URL", "").strip()

# -----------------------------------------------------------------------------
# 공통 유틸
# -----------------------------------------------------------------------------


def norm(s: str) -> str:
    return re.sub(r"\s+", "", (s or "")).strip()


def find_credentials_path():
    for p in [os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
              os.getenv("GSPREAD_CREDENTIALS"),
              "credentials.json", "service_account.json"]:
        if p and os.path.exists(p):
            print(f"[INFO] Using credentials: {p}")
            return p
    content = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if content:
        path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "credentials.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[INFO] Created credentials from env -> {path}")
        return path
    raise FileNotFoundError(
        "서비스 계정 JSON이 필요합니다. (credentials.json 또는 GOOGLE_APPLICATION_CREDENTIALS)")


def open_sheet():
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    creds_file = find_credentials_path()
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_file, scope)
    client = gspread.authorize(creds)

    if not SPREADSHEET_URL:
        raise EnvironmentError("SPREADSHEET_URL이 .env/환경변수에 없습니다.")
    return client.open_by_url(SPREADSHEET_URL)


# -----------------------------------------------------------------------------
# 메모장 GSheet 연동 함수
# -----------------------------------------------------------------------------
MEMO_SHEET_NAME = "메모장"
MEMO_CELL = "A1"


def read_memo_from_gsheet():
    try:
        sh = open_sheet()
        worksheet = sh.worksheet(MEMO_SHEET_NAME)
        content = worksheet.acell(MEMO_CELL).value
        print("[INFO] Memo loaded from GSheet.")
        return content or ""
    except gspread.exceptions.WorksheetNotFound:
        print(
            f"[WARN] Memo sheet '{MEMO_SHEET_NAME}' not found. Returning empty memo.")
        return ""
    except Exception as e:
        print(f"[ERROR] Failed to read memo from GSheet: {e}")
        return f"메모를 불러오는 데 실패했습니다: {e}"


def write_memo_to_gsheet(content: str):
    try:
        sh = open_sheet()
        worksheet = sh.worksheet(MEMO_SHEET_NAME)
        worksheet.update_acell(MEMO_CELL, content)
        print("[INFO] Memo saved to GSheet.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to write memo to GSheet: {e}")
        return False


# -----------------------------------------------------------------------------
# 데이터 로드 (지연 로딩 캐시)
# -----------------------------------------------------------------------------
DATA = {"loaded": False, "daily": pd.DataFrame(), "monthly": pd.DataFrame(),
        "years_options": [], "week_options": []}


def load_data_from_gsheet():
    sh = open_sheet()
    daily_name = os.getenv("DAILY_SHEET_NAME", "일별 발주량 외").strip()
    monthly_name = os.getenv("MONTHLY_SHEET_NAME", "월별 발주량").strip()

    ws_d = sh.worksheet(daily_name)
    # value_render_option='FORMATTED_VALUE'는 셀에 표시된 텍스트를 그대로 가져옵니다.
    vals_d = ws_d.get_all_values(value_render_option='FORMATTED_VALUE')
    di = int(os.getenv("DAILY_HEADER_INDEX", "0"))
    headers_d = [norm(h) for h in vals_d[di]]
    df_daily_raw = pd.DataFrame(vals_d[di+1:], columns=headers_d)

    ws_m = sh.worksheet(monthly_name)
    vals_m = ws_m.get_all_values()
    mi = int(os.getenv("MONTHLY_HEADER_INDEX", "1"))
    headers_m = [norm(h) for h in vals_m[mi]]
    df_monthly_raw = pd.DataFrame(vals_m[mi+1:], columns=headers_m)

    return df_daily_raw, df_monthly_raw

# -----------------------------------------------------------------------------
# 일/월 시트 정제
# -----------------------------------------------------------------------------


def cleanse_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=['날짜', 'date_only', '연도', '월',
                                     '총발주종수', '총발주부수', '흑백페이지', '컬러페이지',
                                     '예상제본시간', '최종출고', '출고부수', '비고', '예상제본시간_시간'])
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
    col_bind = next((c for c in d.columns if c in ['예상제본시간']), None)
    col_shipt = next((c for c in d.columns if c in ['최종출고']), None)
    col_ships = next((c for c in d.columns if c in ['출고부수']), None)
    col_remarks = next((c for c in d.columns if '비고' in c), None)

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
    d['날짜'] = pd.to_datetime(d['__full__'], errors='coerce', yearfirst=True)
    d = d.dropna(subset=['날짜']).copy()

    candidates = [col_cnt, col_total, col_bw, col_color]
    existing = [c for c in candidates if c and c in d.columns]
    if existing:
        blanks = pd.Series(True, index=d.index)
        for c in existing:
            blanks = blanks & d[c].astype(str).str.strip().eq('').fillna(True)
        d = d.loc[~blanks].copy()

    def to_num(series):
        # 쉼표(,)가 포함된 문자열을 숫자로 변환
        s = series.astype(str).str.replace(',', '', regex=False)
        return pd.to_numeric(s, errors='coerce').fillna(0)

    d['총발주종수'] = to_num(d[col_cnt]) if col_cnt and col_cnt in d.columns else 0
    d['총발주부수'] = to_num(
        d[col_total]) if col_total and col_total in d.columns else 0
    d['흑백페이지'] = to_num(d[col_bw]) if col_bw and col_bw in d.columns else 0
    d['컬러페이지'] = to_num(
        d[col_color]) if col_color and col_color in d.columns else 0
    d['출고부수'] = to_num(
        d[col_ships]) if col_ships and col_ships in d.columns else 0

    def as_text(colname):
        if colname and colname in d.columns:
            s = d[colname].astype(str).str.strip()
            s = s.where(~s.eq(''), '-')
            return s
        return '-'

    d['예상제본시간'] = as_text(col_bind)

    # --- 예상 제본 시간 계산 로직 변경 ---
    # 수식을 직접 파이썬 코드로 계산 (결과는 시간 단위)
    # 초 = 837.284 + (9.249 * 총발주종수) + (11.364 * 총발주부수)
    # 시간 = 초 / 3600
    seconds = 837.284 + (9.249 * d['총발주종수']) + (11.364 * d['총발주부수'])
    d['예상제본시간_시간'] = (seconds / 3600).round(2)

    d['최종출고'] = as_text(col_shipt)
    d['비고'] = as_text(col_remarks)

    d['date_only'] = d['날짜'].dt.date
    d['연도'] = d['날짜'].dt.year
    d['월'] = d['날짜'].dt.month

    return d[['날짜', 'date_only', '연도', '월',
              '총발주종수', '총발주부수', '흑백페이지', '컬러페이지',
              '예상제본시간', '예상제본시간_시간', '최종출고', '출고부수', '비고']]


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

# -----------------------------------------------------------------------------
# 영업일/공휴일/월별 가중
# -----------------------------------------------------------------------------


def kr_holidays_for_year(year: int):
    try:
        import holidays
        kr = holidays.KR(years=year)
        return set(kr.keys())
    except Exception as e:
        print(f"[WARN] 'holidays' 사용 불가: {e}. 주말만 제외합니다.")
        return set()


def business_days_in_range(start_d: date, end_d: date, holiday_set: set):
    if end_d < start_d:
        return 0
    cnt = 0
    d = start_d
    while d <= end_d:
        if d.weekday() < 5 and d not in holiday_set:
            cnt += 1
        d += timedelta(days=1)
    return cnt


def get_monthly_business_day_info(today: date):
    year, month = today.year, today.month
    start = date(year, month, 1)
    last_day = calendar.monthrange(year, month)[1]
    end_month = date(year, month, last_day)
    hol = kr_holidays_for_year(year)
    elapsed = business_days_in_range(start, today, hol)
    total = business_days_in_range(start, end_month, hol)
    ratio = (elapsed / total) if total > 0 else 0.0
    return {'elapsed': elapsed, 'total': total, 'ratio': ratio}


def business_day_ratio_for_month(today: date) -> float:
    return get_monthly_business_day_info(today)['ratio']


def total_business_days_in_year(year: int) -> int:
    hol = kr_holidays_for_year(year)
    start = date(year, 1, 1)
    end_year = date(year, 12, 31)
    return business_days_in_range(start, end_year, hol)


def monthly_share_series(df_monthly: pd.DataFrame, year: int, value_col: str) -> pd.Series:
    d = df_monthly[df_monthly['연도'] == year]
    if d.empty or value_col not in d.columns:
        return pd.Series([0.0]*12, index=range(1, 13))
    by_m = d.groupby('월번호')[value_col].sum().reindex(
        range(1, 13), fill_value=0.0)
    total = by_m.sum()
    if total <= 0:
        return pd.Series([0.0]*12, index=range(1, 13))
    return by_m / total


def seasonal_cum_share_to_date(df_monthly: pd.DataFrame, value_col: str, last_year: int, today: date) -> float:
    shares = monthly_share_series(df_monthly, last_year, value_col)
    if shares.sum() == 0:
        return 0.0
    m = today.month
    full_share = shares.loc[1:m-1].sum() if m > 1 else 0.0
    part = shares.loc[m] * business_day_ratio_for_month(today)
    return float(full_share + part)


# -----------------------------------------------------------------------------
# 주간/월별/YoY 그림
# -----------------------------------------------------------------------------
WEEKDAY_KR = ['월', '화', '수', '목', '금', '토', '일']


def last_5_business_days_upto_today(now_kst: datetime):
    dates = []
    d = now_kst.date()
    while len(dates) < 5:
        if d.weekday() < 5:
            dates.append(d)
        d -= timedelta(days=1)
    return list(reversed(dates))


def monday(date_obj: date) -> date:
    return date_obj - timedelta(days=date_obj.weekday())


def week_options_from_df(df_daily: pd.DataFrame):
    if df_daily.empty:
        return [{'label': '(데이터 없음)', 'value': 'this_week'}]
    min_d = df_daily['date_only'].min()
    max_d = df_daily['date_only'].max()
    start_monday = monday(min_d)
    end_monday = monday(max_d)
    opts = []
    cur = end_monday
    while cur >= start_monday:
        label = f"{cur.strftime('%Y-%m-%d')} ~ {(cur + timedelta(days=4)).strftime('%Y-%m-%d')}"
        opts.append({'label': label, 'value': cur.strftime('%Y-%m-%d')})
        cur -= timedelta(days=7)
    return [{'label': '오늘 기준 (최근 5영업일)', 'value': 'this_week'}] + opts


def figure_weekly(df_daily: pd.DataFrame, week_dates: list, value_col: str = '총발주부수', title_prefix=""):
    last_week_dates = [d - timedelta(days=7) for d in week_dates]

    metric_map = {'총발주부수': '발주량', '총발주종수': '발주 종수', '흑백페이지': '흑백 페이지',
                  '컬러페이지': '컬러 페이지', '예상제본시간_시간': '예상 제본 시간'}
    unit_map = {'총발주부수': '부', '총발주종수': '종', '흑백페이지': '페이지',
                '컬러페이지': '페이지', '예상제본시간_시간': '시간'}

    unit = unit_map.get(value_col, '')
    metric_name = metric_map.get(value_col, '값')

    m = df_daily.set_index('date_only')[value_col].to_dict(
    ) if not df_daily.empty and value_col in df_daily.columns else {}

    y_this = [m.get(d, 0) for d in week_dates]
    y_last = [m.get(d, 0) for d in last_week_dates]

    # 4주 이동평균 계산
    y_ma4 = []
    for day in week_dates:
        avg_dates = [day - timedelta(weeks=i)
                     for i in range(1, 5)]  # 1,2,3,4주 전
        vals = [m.get(d, 0) for d in avg_dates]
        y_ma4.append(np.mean(vals) if vals else 0)

    x_week = [WEEKDAY_KR[pd.Timestamp(d).weekday()] for d in week_dates]
    this_dates_str = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in week_dates]
    last_dates_str = [pd.Timestamp(d).strftime('%Y-%m-%d')
                      for d in last_week_dates]

    avg_this = np.mean(y_this) if y_this else 0
    avg_last = np.mean(y_last) if y_last else 0

    fig = go.Figure()

    # 4주 이동평균선
    fig.add_trace(go.Scatter(
        x=x_week, y=y_ma4, mode='lines', name='4주 이동평균',
        line=dict(width=2, dash='longdash', color='rgba(150, 150, 150, 0.7)'),
        hovertemplate="4주 평균: %{y:,.1f}"+unit+"<extra></extra>"
    ))
    # 지난 주
    fig.add_trace(go.Scatter(
        x=x_week, y=y_last, mode='lines+markers+text', name=f'지난 주 (평균: {avg_last:,.1f}{unit})',
        line=dict(width=2, dash='dot'), customdata=last_dates_str,
        hovertemplate="%{customdata}<br>지난 주: %{y:,.1f}" +
        unit+"<extra></extra>",
        text=[f"{v:,.1f}" if v else "" for v in y_last], textposition='top center', textfont={'size': 11}
    ))

    # 변동사항 하이라이트 준비
    marker_colors = ['#1f77b4'] * len(y_this)  # 기본색
    marker_sizes = [8] * len(y_this)
    annotations = []

    for i, (this_val, last_val) in enumerate(zip(y_this, y_last)):
        if last_val > 0:
            change = (this_val - last_val) / last_val
            if abs(change) >= HIGHLIGHT_THRESHOLD:
                marker_colors[i] = '#d62728'  # 강조색
                marker_sizes[i] = 12
                sign = '+' if change > 0 else ''
                annotations.append(dict(
                    x=x_week[i], y=this_val,
                    text=f"{sign}{change:.0%}",
                    showarrow=True, arrowhead=1, ax=0, ay=-25,
                    font=dict(color="white", size=10),
                    bgcolor="#d62728", bordercolor="white", borderwidth=1,
                ))

    # 이번 주
    fig.add_trace(go.Scatter(
        x=x_week, y=y_this, mode='lines+markers+text', name=f'이번 주 (평균: {avg_this:,.1f}{unit})',
        line=dict(width=3), marker=dict(color=marker_colors, size=marker_sizes, line=dict(width=1, color='white')),
        customdata=this_dates_str,
        hovertemplate="%{customdata}<br>이번 주: %{y:,.1f}" +
        unit+"<extra></extra>",
        text=[f"{v:,.1f}" if v else "" for v in y_this], textposition='top center', textfont={'size': 11}
    ))

    fig.update_layout(
        title=f'{title_prefix}주간 {metric_name} 비교',
        xaxis_title='', yaxis_title=f'{metric_name} ({unit})', template='plotly_white', height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation='h', x=1, xanchor='right', y=1.2),
        annotations=annotations
    )
    return fig


def figure_weekly_today_based(df_daily: pd.DataFrame, value_col: str = '총발주부수') -> go.Figure:
    now = datetime.now(KST)
    week_dates = last_5_business_days_upto_today(now)
    title_prefix = f'(기준일: {now.strftime("%Y-%m-%d")}) '
    return figure_weekly(df_daily, week_dates, value_col, title_prefix)


def figure_weekly_fixed_mon_fri(df_daily: pd.DataFrame, monday_str: str = None, value_col: str = '총발주부수') -> go.Figure:
    if monday_str and monday_str != 'this_week':
        try:
            base_mon = datetime.strptime(monday_str, "%Y-%m-%d").date()
        except Exception:
            base_mon = monday(datetime.now(KST).date())
    else:
        base_mon = monday(datetime.now(KST).date())

    week_dates = [base_mon + timedelta(days=i) for i in range(5)]
    title_prefix = f'({week_dates[0].strftime("%m-%d")} ~ {week_dates[-1].strftime("%m-%d")}) '
    return figure_weekly(df_daily, week_dates, value_col, title_prefix)


def figure_months_1to12(df_monthly: pd.DataFrame, start_year=2022, current_year=None) -> go.Figure:
    if current_year is None:
        current_year = datetime.now(KST).year
    d = df_monthly[(df_monthly['연도'] >= start_year) & (
        df_monthly['연도'] <= current_year)].copy()
    if d.empty or '발주량' not in d.columns or d['발주량'].sum() == 0:
        return go.Figure(layout={'title': '월별 발주량 데이터가 없습니다.', 'xaxis': {'visible': False}, 'yaxis': {'visible': False}})
    pivot = d.pivot_table(index='월번호', columns='연도',
                          values='발주량', aggfunc='sum').sort_index()
    pivot = pivot[[c for c in pivot.columns if c <= current_year]]
    fig = px.line(pivot, x=pivot.index, y=pivot.columns, markers=True,
                  title=f'월별 발주량 (1~12월, {start_year}~{current_year})')
    fig.update_layout(xaxis_title='', yaxis_title='발주량', template='plotly_white', height=300,
                      margin=dict(l=20, r=20, t=40, b=20),
                      legend=dict(orientation='h', x=1, xanchor='right', y=1.1))
    fig.update_xaxes(dtick=1)
    return fig


def yoy_line_value_bar_rate(d: pd.DataFrame, value_col: str, title: str, baseline_year: int) -> go.Figure:
    if d.empty or value_col not in d.columns:
        return go.Figure()
    d = d.copy()
    d = d[d['연도'] <= baseline_year]
    d = d.sort_values(['연도', '월번호'])

    d['prev_year'] = d.groupby('월번호')[value_col].shift(1)

    yoy = (d[value_col] - d['prev_year']) / d['prev_year'] * 100

    yoy = yoy.mask(d['prev_year'] == 0, 0)
    yoy = yoy.mask((d[value_col] == 0) & (d['prev_year'] > 0), 0)

    d['YoY%'] = yoy.fillna(0)

    fig = go.Figure()
    for y, sub in d.groupby('연도'):
        fig.add_trace(go.Scatter(
            x=sub['월번호'], y=sub[value_col], mode='lines+markers', name=f'{y}년 ({value_col})'))
    base = d[d['연도'] == baseline_year]
    fig.add_trace(go.Bar(x=base['월번호'], y=base['YoY%'], name=f'{baseline_year} YoY%', yaxis='y2',
                         opacity=0.6, hovertemplate="증감율: %{y:.1f}%<extra></extra>"))
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

# -----------------------------------------------------------------------------
# KPI/진행(참고)
# -----------------------------------------------------------------------------


def compute_kpis(df_daily: pd.DataFrame, year: int):
    if df_daily.empty:
        return 0, 0.0, 0, 0, 0
    year_df = df_daily[df_daily['연도'] == year]
    total_orders = int(year_df['총발주부수'].sum()) if not year_df.empty else 0
    days_count = int(year_df['date_only'].nunique()
                     ) if not year_df.empty else 0
    avg_per_day = round(total_orders / days_count, 2) if days_count else 0.0
    total_bw_pages = int(year_df['흑백페이지'].sum(
    )) if '흑백페이지' in year_df.columns and not year_df.empty else 0
    total_color_pages = int(year_df['컬러페이지'].sum(
    )) if '컬러페이지' in year_df.columns and not year_df.empty else 0
    return total_orders, avg_per_day, days_count, total_bw_pages, total_color_pages


def safe_same_date_last_year(today: date) -> date:
    last_year = today.year - 1
    try:
        return today.replace(year=last_year)
    except ValueError:
        last_day = calendar.monthrange(last_year, today.month)[1]
        return date(last_year, today.month, last_day)


def compute_progress_advanced(df_daily: pd.DataFrame, df_monthly: pd.DataFrame, year: int):
    current_year = datetime.now(KST).year
    today = datetime.now(KST).date()

    if year < current_year:
        as_of_date = date(year, 12, 31)
    else:
        as_of_date = today

    ly = year - 1

    ytd_curr = df_daily[(df_daily['연도'] == year) & (
        df_daily['date_only'] <= as_of_date)]['총발주부수'].sum()

    same_date_ly = safe_same_date_last_year(as_of_date)
    ytd_ly = df_daily[(df_daily['연도'] == ly) & (
        df_daily['date_only'] <= same_date_ly)]['총발주부수'].sum()
    last_year_total = df_daily[df_daily['연도'] == ly]['총발주부수'].sum()

    hol = kr_holidays_for_year(year)
    elapsed_biz_days = business_days_in_range(
        date(year, 1, 1), as_of_date, hol)
    total_biz_days = business_days_in_range(
        date(year, 1, 1), date(year, 12, 31), hol)
    ratio_biz = (elapsed_biz_days / total_biz_days) if total_biz_days else 0.0

    target_biz = last_year_total * ratio_biz if last_year_total else 0
    progress_vs_biz = (ytd_curr / target_biz * 100) if target_biz > 0 else None

    seasonal_share = seasonal_cum_share_to_date(
        df_monthly, '발주량', ly, as_of_date)
    target_seasonal = last_year_total * seasonal_share if last_year_total > 0 else 0
    progress_vs_seasonal = (ytd_curr / target_seasonal *
                            100) if target_seasonal > 0 else None

    return {
        'ytd_curr': int(ytd_curr),
        'ytd_ly': int(ytd_ly),
        'last_year_total': int(last_year_total),
        'ratio_biz': ratio_biz,
        'progress_vs_biz': progress_vs_biz,
        'seasonal_share_to_date': seasonal_share,
        'target_seasonal': int(target_seasonal),
        'target_biz': int(target_biz),
        'progress_vs_seasonal': progress_vs_seasonal,
        'elapsed_biz_days': elapsed_biz_days,
        'total_biz_days': total_biz_days
    }

# -----------------------------------------------------------------------------
# KPI/배지/오늘패널
# -----------------------------------------------------------------------------


def build_weekly_summary_panel(df_daily: pd.DataFrame):
    if df_daily.empty:
        return html.Div("주간 요약 데이터를 불러올 수 없습니다.")

    now = datetime.now(KST)
    week_dates = last_5_business_days_upto_today(now)
    last_week_dates = [d - timedelta(days=7) for d in week_dates]

    df_map = df_daily.set_index('date_only')

    this_week_data = df_map.loc[df_map.index.isin(week_dates)]
    last_week_data = df_map.loc[df_map.index.isin(last_week_dates)]

    if this_week_data.empty:
        return html.Div("이번 주 데이터가 없습니다.")

    this_total = this_week_data['총발주부수'].sum()
    last_total = last_week_data['총발주부수'].sum()

    wow_change = (this_total - last_total) / \
        last_total if last_total > 0 else 0
    wow_str = f"{wow_change:+.1%}"
    wow_color = "#2b8a3e" if wow_change >= 0 else "#d9480f"

    this_avg = this_week_data['총발주부수'].mean()

    peak_day_row = this_week_data.loc[this_week_data['총발주부수'].idxmax()]
    peak_day_val = peak_day_row['총발주부수']
    peak_day_date = peak_day_row.name
    peak_day_str = f"{WEEKDAY_KR[peak_day_date.weekday()]}요일"

    style_p = {'margin': '2px 0', 'fontSize': '0.9rem'}
    style_strong = {'margin': '0 4px'}

    summary_text = [
        html.P([
            "금주 총 발주량은 ",
            html.Strong(f"{this_total:,.0f}부", style=style_strong),
            "로, 지난주 대비 ",
            html.Strong(f"{wow_str}", style={
                        **style_strong, 'color': wow_color}),
            " 변동했습니다."
        ], style=style_p),
        html.P([
            "일 평균 ",
            html.Strong(f"{this_avg:,.1f}부", style=style_strong),
            "를 기록했으며, ",
            html.Strong(f"{peak_day_str}({peak_day_val:,.0f}부)",
                        style=style_strong),
            "에 가장 많은 발주가 있었습니다."
        ], style=style_p)
    ]

    return html.Div(summary_text, style={
        'background': '#f8f9fa', 'borderRadius': '8px', 'padding': '10px 14px',
        'border': '1px solid #e9ecef', 'marginBottom': '10px'
    })


def badge(text, color="#2b8a3e", tip=None):
    return html.Span(
        text,
        title=tip,
        style={
            'display': 'inline-block', 'padding': '4px 8px', 'borderRadius': '999px',
            'background': color, 'color': 'white', 'fontSize': '0.8rem', 'fontWeight': '700', 'cursor': 'help'
        }
    )


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


def build_kpi_layout(df_daily: pd.DataFrame, df_monthly: pd.DataFrame, year: int):
    tot, avg, days, bw, color = compute_kpis(df_daily, year)
    row1 = html.Div(style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap', 'marginBottom': '8px'}, children=[
        kpi_card(f"{year}년 총 발주량", f"{tot:,}", "Total Orders"),
        kpi_card(f"{year}년 일 평균 발주량", f"{avg:,}", "Avg / Working Day"),
        kpi_card(f"{year}년 총 발주 일수", f"{days:,}일", "Working Days Count"),
    ])
    row2 = html.Div(style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap', 'marginBottom': '6px'}, children=[
        kpi_card(f"{year}년 흑백 페이지 합계", f"{bw:,}", "BW Pages (Daily sum)"),
        kpi_card(f"{year}년 컬러 페이지 합계", f"{color:,}",
                 "Color Pages (Daily sum)"),
    ])

    prog = compute_progress_advanced(df_daily, df_monthly, year)
    today = datetime.now(KST).date()
    monthly_info = get_monthly_business_day_info(today)

    badges = []

    if prog['ytd_ly'] > 0:
        ytd_vs_ly = prog['ytd_curr'] / prog['ytd_ly'] * 100
        diff = ytd_vs_ly - 100
        interpretation = f"작년 동기간 대비 {abs(diff):.1f}% 발주량이 {'늘었어요' if diff >= 0 else '줄었어요'}!"
        calculation = f"산출 근거: (올해 YTD {prog['ytd_curr']:,} ÷ 작년 YTD {prog['ytd_ly']:,}) × 100"
        tip = f"{interpretation}\n{'-'*20}\n{calculation}"
        badges.append(
            badge(f"YTD vs 작년 동기간: {ytd_vs_ly:.1f}%", "#0b7285", tip=tip))

    if prog['progress_vs_biz'] is not None:
        diff = prog['progress_vs_biz'] - 100
        interpretation = f"단순 영업일 경과율 기준 목표보다 {abs(diff):.1f}% {'빠르게' if diff >= 0 else '느리게'} 진행중이에요!"
        calculation = (f"산출 근거: {prog['ytd_curr']:,} (올해 YTD) ÷ \n"
                       f"({prog['last_year_total']:,} (작년 총량) × {prog['ratio_biz']:.1%} (진행율) = {prog['target_biz']:,})")
        tip = f"{interpretation}\n{'-'*20}\n{calculation}"
        badges.append(badge(f"경과율(영업일) 대비: {prog['progress_vs_biz']:.1f}%",
                      "#2b8a3e" if prog['progress_vs_biz'] >= 100 else "#d9480f", tip=tip))

    if prog['progress_vs_seasonal'] is not None:
        diff = prog['progress_vs_seasonal'] - 100
        interpretation = f"과거 월별 패턴(계절성) 기준 목표보다 {abs(diff):.1f}% {'앞서가고' if diff >= 0 else '뒤쳐지고'} 있어요!"
        calculation = f"산출 근거: {prog['ytd_curr']:,} (올해 YTD) ÷ {prog['target_seasonal']:,} (작년 패턴 기준 목표)"
        tip = f"{interpretation}\n{'-'*20}\n{calculation}"
        badges.append(badge(f"월별 가중치 대비: {prog['progress_vs_seasonal']:.1f}%",
                      "#2b8a3e" if prog['progress_vs_seasonal'] >= 100 else "#d9480f", tip=tip))

    interpretation_annual = f"{year}년 전체 영업일({prog['total_biz_days']}일) 중 {prog['elapsed_biz_days']}일이 지났어요."
    calculation_annual = f"산출 근거: {prog['elapsed_biz_days']}일 ÷ {prog['total_biz_days']}일"
    tip_annual = f"{interpretation_annual}\n{'-'*20}\n{calculation_annual}"
    badges.append(
        badge(f"연간 영업일 경과율: {prog['ratio_biz']*100:.1f}%", "#5f3dc4", tip=tip_annual))

    interpretation_monthly = f"이번 달(현재 {today.month}월) 전체 영업일({monthly_info['total']}일) 중 {monthly_info['elapsed']}일이 지났어요."
    calculation_monthly = f"산출 근거: {monthly_info['elapsed']}일 ÷ {monthly_info['total']}일"
    tip_monthly = f"{interpretation_monthly}\n{'-'*20}\n{calculation_monthly}"
    badges.append(badge(
        f"월간 영업일 경과율: {monthly_info['ratio']*100:.1f}%", "#e67700", tip=tip_monthly))

    row3 = html.Div(style={'display': 'flex', 'gap': '8px', 'flexWrap': 'wrap',
                    'alignItems': 'center', 'margin': '4px 2px 0'}, children=badges)
    return html.Div(children=[row1, row2, row3])

# ===== 오늘 현황 패널 =====


def build_today_panel(df_daily: pd.DataFrame):
    today = datetime.now(KST).date()
    if df_daily.empty or 'date_only' not in df_daily.columns:
        body = html.Div("데이터를 불러오는 중이거나, 오늘 데이터가 없습니다.",
                        style={'color': '#666'})
    else:
        row = df_daily[df_daily['date_only'] == today]
        if row.empty:
            body = html.Div([html.Div("오늘 데이터가 아직 없습니다.", style={
                            'color': '#666', 'marginBottom': '6px'})])
        else:
            r = row.iloc[0]

            grid = html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'rowGap': '6px', 'columnGap': '8px'}, children=[
                html.Div("총 발주 종수", style={'color': '#666'}), html.Div(
                    f"{int(r.get('총발주종수', 0)):,}", style={'textAlign': 'right', 'fontWeight': '700'}),
                html.Div("총 발주 부수", style={'color': '#666'}), html.Div(
                    f"{int(r.get('총발주부수', 0)):,}",   style={'textAlign': 'right', 'fontWeight': '700'}),
                html.Div("흑백 페이지",  style={'color': '#666'}), html.Div(
                    f"{int(r.get('흑백페이지', 0)):,}",    style={'textAlign': 'right', 'fontWeight': '700'}),
                html.Div("컬러 페이지",  style={'color': '#666'}), html.Div(
                    f"{int(r.get('컬러페이지', 0)):,}", style={'textAlign': 'right', 'fontWeight': '700'}),
                # --- 오늘 현황 패널 표시 방식 변경 ---
                # 시트의 표시된 텍스트를 그대로 사용
                html.Div("예상 제본 시간", style={'color': '#666'}), html.Div(
                    (str(r.get('예상제본시간', '-')) or '-'),  style={'textAlign': 'right', 'fontWeight': '700'}),
                html.Div("최종 출고",    style={'color': '#666'}), html.Div(
                    (str(r.get('최종출고', '-')) or '-'),   style={'textAlign': 'right', 'fontWeight': '700'}),
                html.Div("출고 부수",    style={'color': '#666'}), html.Div(
                    f"{int(r.get('출고부수', 0)):,}",    style={'textAlign': 'right', 'fontWeight': '700'}),
            ])

            remarks = (str(r.get('비고', '-')) or '-')
            if remarks and remarks != '-':
                remarks_section = html.Div([
                    html.Hr(style={'margin': '10px 0',
                            'borderColor': '#e9ecef'}),
                    html.Div("비고", style={
                             'color': '#666', 'fontWeight': '700', 'marginBottom': '4px'}),
                    html.Div(remarks, style={
                             'whiteSpace': 'pre-wrap', 'fontSize': '0.85rem', 'lineHeight': '1.5', 'wordBreak': 'break-all'})
                ])
                body = html.Div([grid, remarks_section])
            else:
                body = grid

    header = html.Div([
        html.Div("오늘 현황", style={'fontWeight': '800', 'fontSize': '1.05rem'}),
        html.Div(datetime.now(KST).strftime("%Y-%m-%d (%a) %H:%M"),
                 style={'color': '#888', 'fontSize': '0.8rem', 'marginTop': '2px'})
    ])
    return html.Div(style={'position': 'fixed', 'top': '80px', 'right': '16px', 'width': 'min(94vw, 310px)', 'zIndex': '999', 'background': 'rgba(255,255,255,0.98)', 'backdropFilter': 'blur(2px)', 'border': '1px solid #edf2f7', 'borderRadius': '14px', 'padding': '12px 14px', 'boxShadow': '0 10px 22px rgba(0,0,0,0.12)'}, children=[header, html.Hr(style={'margin': '8px 0', 'borderColor': '#f1f3f5'}), body])

# -----------------------------------------------------------------------------
# 예측 섹션
# -----------------------------------------------------------------------------


def compute_advanced_forecasts(df_daily: pd.DataFrame, year: int):
    today = datetime.now(KST).date()
    last_year = year - 1
    metrics = ['총발주부수', '흑백페이지', '컬러페이지']
    forecasts = {}

    ytd_daily_df = df_daily[(df_daily['연도'] == year) &
                            (df_daily['date_only'] <= today)]
    if not ytd_daily_df.empty:
        elapsed_days = ytd_daily_df['date_only'].nunique()
        ytd_total_orders = ytd_daily_df['총발주부수'].sum()
        forecasts['일 평균 발주량'] = round(
            ytd_total_orders / elapsed_days, 1) if elapsed_days > 0 else 0.0
    else:
        forecasts['일 평균 발주량'] = 0.0

    for metric in metrics:
        current_ytd = df_daily[(df_daily['연도'] == year) & (
            df_daily['date_only'] <= today)][metric].sum()

        same_date_ly = safe_same_date_last_year(today)
        last_year_ytd = df_daily[(df_daily['연도'] == last_year) & (
            df_daily['date_only'] <= same_date_ly)][metric].sum()

        last_year_remaining = df_daily[(df_daily['연도'] == last_year) & (
            df_daily['date_only'] > same_date_ly)][metric].sum()

        growth_rate = current_ytd / last_year_ytd if last_year_ytd > 0 else 1.0

        forecast_remaining = last_year_remaining * growth_rate

        total_forecast = current_ytd + forecast_remaining
        forecasts[metric] = int(round(total_forecast))

    forecasts['총 발주량'] = forecasts.pop('총발주부수')
    forecasts['흑백 페이지 합계'] = forecasts.pop('흑백페이지')
    forecasts['컬러 페이지 합계'] = forecasts.pop('컬러페이지')

    return forecasts


def forecast_cards_layout(df_daily, df_monthly, year: int):
    fx = compute_advanced_forecasts(df_daily, year)

    def kv_row(k, v, hint=""):
        if v is None:
            val = "-"
        elif isinstance(v, int):
            val = f"{v:,}"
        else:
            val = f"{v:,.1f}"

        sub = html.Span(
            hint, style={'color': '#8a8a8a', 'fontSize': '0.8rem'}) if hint else None
        return html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr auto', 'alignItems': 'baseline', 'padding': '8px 0', 'borderBottom': '1px solid #f1f3f5'}, children=[
            html.Div([html.Strong(k), html.Span(" "), sub]),
            html.Div(val, style={'fontWeight': '800', 'fontSize': '1.1rem'})
        ])

    header = html.Div([
        html.Div(f"{year}년 예상치 (실시간 실적 및 계절성 반영)", style={
                 'fontWeight': '800', 'fontSize': '1.05rem', 'marginBottom': '2px'}),
        html.Div("YTD 실적 + (YoY 성장률 × 작년 잔여 실적)으로 연말까지의 성과를 예측합니다.",
                 style={'color': '#888', 'fontSize': '0.85rem'})
    ], style={'marginBottom': '8px'})

    grid = html.Div(style={'background': 'white', 'borderRadius': '12px', 'padding': '12px', 'boxShadow': '0 4px 14px rgba(0,0,0,0.08)'}, children=[
        kv_row("총 발주량",       fx.get('총 발주량'),       "누적+예측"),
        kv_row("일 평균 발주량",   fx.get('일 평균 발주량'),   "YTD 실적 기준"),
        kv_row("흑백 페이지 합계", fx.get('흑백 페이지 합계'), "누적+예측"),
        kv_row("컬러 페이지 합계", fx.get('컬러 페이지 합계'), "누적+예측"),
    ])
    return html.Div(children=[header, grid])


# -----------------------------------------------------------------------------
# 앱 (지연 로딩 레이아웃)
# -----------------------------------------------------------------------------
external_stylesheets = [
    {"href": "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css", "rel": "stylesheet"}]
app = dash.Dash(__name__,
                title="발주량 분석 대시보드",
                meta_tags=[{"name": "viewport",
                            "content": "width=device-width, initial-scale=1"}],
                external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True)
server = app.server
CURRENT_YEAR = datetime.now(KST).year

# -----------------------------------------------------------------------------
# 메인 대시보드 레이아웃 함수
# -----------------------------------------------------------------------------


def create_main_dashboard_layout():
    return html.Div(style={'maxWidth': '1100px', 'margin': '0 auto', 'padding': '16px', 'fontFamily': 'Noto Sans KR, Malgun Gothic, Arial'}, children=[
        dcc.Interval(id='init', interval=250, n_intervals=0, max_intervals=1),
        dcc.Interval(id='today-refresh', interval=120*1000, n_intervals=0),
        dcc.Interval(id='data-auto-refresh',
                     interval=AUTO_REFRESH_MIN*60*1000, n_intervals=0),

        dcc.Store(id='data-version', storage_type='memory', data=0),
        dcc.Store(id='today-visible', storage_type='local', data=True),
        dcc.Store(id='memo-visible', storage_type='local', data=False),

        html.H1("발주량 분석 대시보드", style={
                'textAlign': 'center', 'marginBottom': '6px'}),
        html.P("구글 시트 데이터 기반 · 2021~현재", style={
               'textAlign': 'center', 'marginBottom': '14px', 'color': '#666'}),

        html.Div(style={'display': 'flex', 'gap': '10px', 'justifyContent': 'center', 'alignItems': 'center', 'marginBottom': '8px', 'flexWrap': 'wrap'}, children=[
            html.Span("KPI 연도 선택:", style={'fontWeight': '600'}),
            dcc.Dropdown(id='year-select', options=[{'label': f'{CURRENT_YEAR}년', 'value': CURRENT_YEAR}],
                         value=CURRENT_YEAR, clearable=False, style={'width': '180px'}),
            html.Button("데이터 새로고침", id="manual-refresh", n_clicks=0, title="구글 시트에서 다시 불러오기", style={
                        'borderRadius': '8px', 'padding': '8px 12px', 'fontWeight': '700', 'border': '1px solid #e2e8f0', 'background': '#2563eb', 'color': 'white', 'cursor': 'pointer'}),
            html.A(html.Button("원본 시트 바로가기"), href=SPREADSHEET_URL or "#", target="_blank",
                   style={'textDecoration': 'none'}),
            dcc.Link(html.Button('청구액 페이지'), href='/billing',
                     style={'textDecoration': 'none'}),
            html.Div(id='kpi-refresh-status',
                     style={'marginLeft': '12px', 'color': '#888'})
        ]),

        html.Div(id='kpi-cards'),

        html.Details(open=False, children=[
            html.Summary("지난 연도 KPI 펼치기 / 접기"),
            html.Div(id='prev-years-kpi', style={'marginTop': '8px'})
        ], style={'background': 'white', 'borderRadius': '12px', 'padding': '14px', 'boxShadow': '0 4px 14px rgba(0,0,0,0.08)', 'marginBottom': '16px'}),

        html.Div(style={'background': 'white', 'borderRadius': '12px', 'padding': '14px', 'boxShadow': '0 4px 14px rgba(0,0,0,0.08)', 'marginBottom': '12px'}, children=[
            html.H3("주간 비교 분석", style={
                    'marginBottom': '8px', 'fontSize': '1.05rem'}),
            html.Div(id='weekly-summary-container'),
            dcc.Tabs(id="weekly-tabs-today", value="총발주부수", children=[
                dcc.Tab(label="발주량", value="총발주부수"),
                dcc.Tab(label="발주 종수", value="총발주종수"),
                dcc.Tab(label="예상 제본 시간", value="예상제본시간_시간"),
                dcc.Tab(label="흑백 페이지", value="흑백페이지"),
                dcc.Tab(label="컬러 페이지", value="컬러페이지"),
            ]),
            dcc.Graph(id='weekly-chart-today', figure=go.Figure(),
                      style={'height': '340px'}),
        ]),

        html.Details(open=False, children=[
            html.Summary("월~금 고정 주간 비교 (클릭하여 열기)"),
            html.Div(style={'display': 'flex', 'gap': '8px', 'alignItems': 'center', 'margin': '8px 0'}, children=[
                html.Span("주차 선택:", style={'fontWeight': '600'}),
                dcc.Dropdown(id='week-select-fixed', options=[
                             {'label': '오늘 기준 (최근 5영업일)', 'value': 'this_week'}], value='this_week', clearable=False, style={'width': '300px'}),
            ]),
            dcc.Tabs(id="weekly-tabs-fixed", value="총발주부수", children=[
                dcc.Tab(label="발주량", value="총발주부수"),
                dcc.Tab(label="발주 종수", value="총발주종수"),
                dcc.Tab(label="예상 제본 시간", value="예상제본시간_시간"),
                dcc.Tab(label="흑백 페이지", value="흑백페이지"),
                dcc.Tab(label="컬러 페이지", value="컬러페이지"),
            ]),
            dcc.Graph(id='weekly-chart-fixed', style={'height': '340px'})
        ], style={'background': 'white', 'borderRadius': '12px', 'padding': '14px', 'boxShadow': '0 4px 14px rgba(0,0,0,0.08)', 'marginBottom': '12px'}),

        html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr', 'gap': '12px', 'marginBottom': '12px'}, children=[
            html.Div([html.H3("월별 발주량 (1~12월, 2022~현재)", style={'marginBottom': '4px', 'fontSize': '1.05rem'}),
                      dcc.Graph(id='months-1to12-chart', figure=go.Figure(), style={'height': '320px'})],
                     style={'background': 'white', 'borderRadius': '12px', 'padding': '14px', 'boxShadow': '0 4px 14px rgba(0,0,0,0.08)'}),
        ]),

        html.Div(style={'background': 'white', 'borderRadius': '12px', 'padding': '0px', 'boxShadow': '0 4px 14px rgba(0,0,0,0.08)'}, children=[
            dcc.Tabs(id="metric-tabs", value="avg", children=[
                dcc.Tab(label="일평균 발주량", value="avg"),
                dcc.Tab(label="월 총 발주량", value="total"),
                dcc.Tab(label="흑백 페이지", value="bw"),
                dcc.Tab(label="컬러 페이지", value="color"),
                dcc.Tab(label="연간 예측", value="forecast"),
            ]),
            html.Div(id="metric-tab-content", style={'padding': '8px 4px'})
        ]),

        html.Div(id='today-floating-panel'),

        html.Div(id='memo-popup', style={'position': 'fixed', 'left': '16px', 'top': '80px', 'width': 'min(92vw, 360px)', 'zIndex': '1000', 'display': 'none', 'background': 'rgba(255,255,255,0.98)', 'backdropFilter': 'blur(2px)', 'border': '1px solid #edf2f7', 'borderRadius': '14px', 'padding': '10px 12px', 'boxShadow': '0 10px 22px rgba(0,0,0,0.12)', 'maxHeight': '70vh', 'overflowY': 'auto'}, children=[
            html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '6px'}, children=[
                html.Div("메모장 (구글 시트 연동)", style={
                         'fontWeight': '800', 'fontSize': '1.0rem'}),
                html.Button("닫기", id="memo-close-btn", n_clicks=0)
            ]),
            html.Div(style={'display': 'flex', 'gap': '6px', 'marginBottom': '8px', 'flexWrap': 'wrap'}, children=[
                html.Button("타임스탬프", id="memo-timestamp-btn", n_clicks=0),
                html.Button("저장", id="memo-save-btn", n_clicks=0,
                            style={'background': '#2563eb', 'color': 'white', 'border': 'none'}),
                html.Button("지우기", id="memo-clear-btn", n_clicks=0),
                html.Div([
                    dcc.Clipboard(id="memo-clipboard",
                                  style={"display": "inline"}),
                    html.Button("클립보드 복사", id="memo-copy-btn", n_clicks=0),
                ]),
            ]),
            dcc.Textarea(
                id='memo-text', style={'width': '100%', 'height': '30vh', 'resize': 'vertical'}, spellCheck=False),
            html.Div(style={'marginTop': '6px', 'textAlign': 'right',
                     'color': '#888', 'fontSize': '0.8rem'}, id='memo-status')
        ]),

        html.Button("오늘 현황", id="fab-today-toggle", n_clicks=0, title="오른쪽 패널 열기/닫기", style={'position': 'fixed', 'right': '16px', 'bottom': '16px', 'borderRadius': '999px', 'padding': '12px 16px',
                    'fontWeight': '800', 'background': '#2563eb', 'color': 'white', 'border': 'none', 'boxShadow': '0 8px 18px rgba(0,0,0,0.18)', 'zIndex': 1100, 'cursor': 'pointer'}),
        html.Button("메모장", id="fab-memo-toggle", n_clicks=0, title="메모장 열기/닫기", style={'position': 'fixed', 'left': '16px', 'bottom': '16px', 'borderRadius': '999px', 'padding': '12px 16px',
                    'fontWeight': '800', 'background': '#059669', 'color': 'white', 'border': 'none', 'boxShadow': '0 8px 18px rgba(0,0,0,0.18)', 'zIndex': 1100, 'cursor': 'pointer'}),
    ])


# -----------------------------------------------------------------------------
# 앱 전체 레이아웃 (라우팅)
# -----------------------------------------------------------------------------
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


@callback(Output('page-content', 'children'),
          Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/billing':
        return billing_page.create_layout()
    else:
        return create_main_dashboard_layout()

# -----------------------------------------------------------------------------
# 데이터 로딩/리프레시
# -----------------------------------------------------------------------------


def ensure_data_loaded():
    if DATA["loaded"]:
        return
    try:
        raw_d, raw_m = load_data_from_gsheet()
        df_d = cleanse_daily(raw_d)
        df_m = cleanse_monthly(raw_m)
        DATA["daily"] = df_d
        DATA["monthly"] = df_m
        if not df_d.empty:
            years = sorted(df_d['연도'].unique())
            DATA["years_options"] = [
                {'label': f'{int(y)}년', 'value': int(y)} for y in years]
            DATA["week_options"] = week_options_from_df(df_d)
        else:
            DATA["years_options"] = [
                {'label': f'{CURRENT_YEAR}년', 'value': CURRENT_YEAR}]
            DATA["week_options"] = [
                {'label': '오늘 기준 (최근 5영업일)', 'value': 'this_week'}]
        DATA["loaded"] = True
        print("[INFO] Data loaded successfully.")
    except Exception as e:
        print("[ERROR] 데이터 로드 실패:", e)
        DATA["daily"], DATA["monthly"] = pd.DataFrame(), pd.DataFrame()
        DATA["years_options"] = [
            {'label': f'{CURRENT_YEAR}년', 'value': CURRENT_YEAR}]
        DATA["week_options"] = [
            {'label': '오늘 기준 (최근 5영업일)', 'value': 'this_week'}]
        DATA["loaded"] = True


@callback(
    Output('data-version', 'data'),
    Input('data-auto-refresh', 'n_intervals'),
    Input('manual-refresh', 'n_clicks'),
    State('data-version', 'data'),
    prevent_initial_call=False
)
def auto_or_manual_reload(_auto_n, _manual_clicks, ver):
    try:
        print("Reloading data...")
        DATA["loaded"] = False
        ensure_data_loaded()
        return (ver or 0) + 1
    except Exception as e:
        print(f"Data reload failed: {e}")
        raise PreventUpdate

# -----------------------------------------------------------------------------
# 콜백
# -----------------------------------------------------------------------------


@callback(Output('year-select', 'options'), Output('year-select', 'value'), Input('init', 'n_intervals'), Input('data-version', 'data'), prevent_initial_call=False)
def init_year_options(_, _ver):
    ensure_data_loaded()
    opts = DATA["years_options"]
    latest = max([o['value'] for o in opts]) if opts else CURRENT_YEAR
    return opts, latest


@callback(Output('kpi-cards', 'children'), Output('prev-years-kpi', 'children'), Output('kpi-refresh-status', 'children'), Input('year-select', 'value'), Input('data-version', 'data'))
def update_kpis(selected_year, _ver):
    ensure_data_loaded()
    df_d, df_m = DATA["daily"], DATA["monthly"]
    try:
        kpi_layout = build_kpi_layout(df_d, df_m, selected_year)
        if df_d.empty:
            prev_tbl = html.Div("(데이터 없음)")
        else:
            years = sorted(df_d['연도'].unique())
            td_style = {'textAlign': 'right', 'padding': '6px 8px', 'borderBottom': '1px solid #f1f3f5',
                        'fontVariantNumeric': 'tabular-nums', 'whiteSpace': 'nowrap'}
            th_style = {'textAlign': 'right', 'padding': '6px 8px',
                        'borderBottom': '2px solid #dee2e6', 'color': '#495057'}
            rows = [html.Tr([html.Td(f"{y}년", style={**td_style, 'textAlign': 'left'})] + [html.Td(
                f"{val:,}", style=td_style) for val in compute_kpis(df_d, y)[:3]]) for y in years if y != selected_year]
            header = html.Tr([html.Th(h, style={**th_style, 'textAlign': 'left' if i == 0 else 'right'})
                             for i, h in enumerate(["연도", "총 발주량", "일 평균 발주량", "총 발주 일수"])])
            prev_tbl = html.Table([header] + rows, style={
                                  'width': '100%', 'borderCollapse': 'collapse', 'tableLayout': 'fixed'}, className="kpi-table")
        stamp = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")
        return kpi_layout, prev_tbl, f"업데이트: {stamp}"
    except Exception as e:
        return html.Div(f"KPI 계산 오류: {e}"), html.Div(f"오류: {e}"), ""


@callback(Output('week-select-fixed', 'options'), Output('week-select-fixed', 'value'), Input('data-version', 'data'))
def refresh_week_options(_ver):
    ensure_data_loaded()
    return DATA["week_options"], 'this_week'


@callback(Output('weekly-summary-container', 'children'), Input('data-version', 'data'))
def update_weekly_summary(_ver):
    ensure_data_loaded()
    return build_weekly_summary_panel(DATA["daily"])


@callback(Output('weekly-chart-today', 'figure'), Input('weekly-tabs-today', 'value'), Input('data-version', 'data'))
def refresh_weekly_today(selected_metric, _ver):
    ensure_data_loaded()
    return figure_weekly_today_based(DATA["daily"], value_col=selected_metric)


@callback(Output('weekly-chart-fixed', 'figure'), Input('week-select-fixed', 'value'), Input('weekly-tabs-fixed', 'value'), Input('data-version', 'data'))
def update_week_fixed(monday_str, selected_metric, _ver):
    ensure_data_loaded()
    return figure_weekly_fixed_mon_fri(DATA["daily"], monday_str, value_col=selected_metric)


@callback(Output('months-1to12-chart', 'figure'), Input('year-select', 'value'), Input('data-version', 'data'))
def refresh_month_chart(_selected_year, _ver):
    ensure_data_loaded()
    cy = datetime.now(KST).year
    return figure_months_1to12(DATA["monthly"], start_year=2022, current_year=cy)


@callback(Output("metric-tab-content", "children"), Input("metric-tabs", "value"), Input("year-select", "value"), Input("data-version", "data"))
def switch_metric_tab(tab_value, selected_year, _ver):
    ensure_data_loaded()
    if tab_value == "forecast":
        return html.Div(style={'padding': '10px'}, children=[forecast_cards_layout(DATA["daily"], DATA["monthly"], selected_year)])

    col_map = {"avg": "일평균발주량", "total": "발주량",
               "bw": "흑백출력량", "color": "컬러출력량"}
    title_map = {"avg": "월별 일평균 발주량", "total": "월 총 발주량",
                 "bw": "월별 흑백 페이지", "color": "월별 컬러 페이지"}
    fig = yoy_line_value_bar_rate(
        DATA["monthly"], col_map[tab_value], f'{title_map[tab_value]} + YoY%', selected_year)
    fig.update_layout(height=460)
    return dcc.Graph(figure=fig, style={'height': '480px'})


@callback(Output('today-floating-panel', 'children'), Input('today-refresh', 'n_intervals'), Input('data-version', 'data'), prevent_initial_call=False)
def refresh_today_panel(_n, _ver):
    ensure_data_loaded()
    return build_today_panel(DATA["daily"])


@callback(Output('today-floating-panel', 'style'), Input('today-visible', 'data'), prevent_initial_call=False)
def apply_today_visibility(visible):
    return {} if bool(visible) or visible is None else {'display': 'none'}


@callback(Output('today-visible', 'data'), Input('fab-today-toggle', 'n_clicks'), State('today-visible', 'data'), prevent_initial_call=True)
def toggle_today_visible(_n, visible):
    return not bool(visible)

# -----------------------------------------------------------------------------
# 메모장 콜백
# -----------------------------------------------------------------------------


clientside_callback(
    """
    function(n_clicks, existing_text) {
        if (n_clicks === null || n_clicks === undefined || n_clicks === 0) {
            return dash_clientside.no_update;
        }
        const now = new Date();
        const year = now.getFullYear();
        const month = String(now.getMonth() + 1).padStart(2, '0');
        const day = String(now.getDate()).padStart(2, '0');
        const hours = String(now.getHours()).padStart(2, '0');
        const minutes = String(now.getMinutes()).padStart(2, '0');
        const timestamp = `${year}-${month}-${day} ${hours}:${minutes}`;
        
        if (existing_text) {
            return existing_text + '\\n' + timestamp + ': ';
        }
        return timestamp + ': ';
    }
    """,
    Output('memo-text', 'value', allow_duplicate=True),
    Input('memo-timestamp-btn', 'n_clicks'),
    State('memo-text', 'value'),
    prevent_initial_call=True
)


@callback(
    Output('memo-clipboard', 'content'),
    Input('memo-copy-btn', 'n_clicks'),
    State('memo-text', 'value'),
    prevent_initial_call=True
)
def copy_memo_to_clipboard(n_clicks, text):
    return text or ""


@callback(
    Output('memo-visible', 'data'),
    Input('fab-memo-toggle', 'n_clicks'),
    Input('memo-close-btn', 'n_clicks'),
    State('memo-visible', 'data'),
    prevent_initial_call=True
)
def toggle_memo_visible(fab_clicks, close_clicks, visible):
    ctx = dash.callback_context
    if not ctx.triggered:
        return visible
    if ctx.triggered_id == 'memo-close-btn':
        return False
    return not bool(visible)


@callback(
    Output('memo-popup', 'style'),
    Output('memo-text', 'value'),
    Output('memo-status', 'children'),
    Input('memo-visible', 'data'),
    Input('memo-save-btn', 'n_clicks'),
    Input('memo-clear-btn', 'n_clicks'),
    State('memo-text', 'value'),
    prevent_initial_call=True
)
def memo_controller(visible, save_n, clear_n, text):
    base_style = {'position': 'fixed', 'left': '16px', 'top': '80px', 'width': 'min(92vw, 360px)', 'zIndex': '1000', 'background': 'rgba(255,255,255,0.98)', 'backdropFilter': 'blur(2px)',
                  'border': '1px solid #edf2f7', 'borderRadius': '14px', 'padding': '10px 12px', 'boxShadow': '0 10px 22px rgba(0,0,0,0.12)', 'maxHeight': '70vh', 'overflowY': 'auto'}
    display_style = {'display': 'block' if visible else 'none'}

    ctx = dash.callback_context
    triggered_id = ctx.triggered_id

    if triggered_id == 'memo-visible' and visible:
        content_from_sheet = read_memo_from_gsheet()
        return {**base_style, **display_style}, content_from_sheet, "로드 완료"

    if triggered_id == 'memo-save-btn':
        success = write_memo_to_gsheet(text or "")
        status = "저장 완료" if success else "오류: 저장 실패"
        return {**base_style, **display_style}, text, status

    if triggered_id == 'memo-clear-btn':
        success = write_memo_to_gsheet("")
        status = "삭제 완료" if success else "오류: 삭제 실패"
        new_text = "" if success else text
        return {**base_style, **display_style}, new_text, status

    if not visible:
        return {**base_style, 'display': 'none'}, dash.no_update, ""

    return {**base_style, **display_style}, dash.no_update, ""


# -----------------------------------------------------------------------------
# 로컬 실행
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    host = os.getenv('DASH_HOST', '127.0.0.1')
    port = int(os.getenv('DASH_PORT', '8090'))
    print(f"브라우저에서 http://{host}:{port} 으로 접속하세요.")
    app.run(debug=True, host=host, port=port)

import os
import re
import calendar
from datetime import datetime, timedelta, date

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

# -----------------------------
# Utils
# -----------------------------


def norm(s: str) -> str:
    return re.sub(r"\s+", "", (s or "")).strip()


def find_credentials_path():
    for p in [os.getenv("GOOGLE_APPLICATION_CREDENTIALS"), os.getenv("GSPREAD_CREDENTIALS"), "credentials.json", "service_account.json"]:
        if p and os.path.exists(p):
            print(f"[INFO] Using credentials: {p}")
            return p
    raise FileNotFoundError("서비스 계정 JSON이 필요합니다. (credentials.json 또는 env 경로)")


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

# -----------------------------
# Load
# -----------------------------


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

    return df_daily_raw, df_monthly_raw

# -----------------------------
# Cleanse
# -----------------------------


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
    d['날짜'] = pd.to_datetime(d['__full__'], errors='coerce', yearfirst=True)
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

# -----------------------------
# Business days & holiday logic
# -----------------------------


def kr_holidays_for_year(year: int):
    """Return a set of KR public holidays for the year. Falls back to empty set if 'holidays' not installed."""
    try:
        import holidays  # type: ignore
        kr = holidays.KR(years=year)
        return set(kr.keys())
    except Exception as e:
        print(f"[WARN] 'holidays' 패키지를 사용할 수 없습니다: {e}. 주말만 제외하여 영업일 계산합니다.")
        return set()


def business_days_in_range(start_d: date, end_d: date, holiday_set: set):
    """Count business days between two dates inclusive, excluding weekends and holidays."""
    if end_d < start_d:
        return 0
    cnt = 0
    d = start_d
    while d <= end_d:
        if d.weekday() < 5 and d not in holiday_set:
            cnt += 1
        d += timedelta(days=1)
    return cnt


def business_day_ratio_for_year(today: date) -> float:
    """Elapsed business days / full-year business days (Mon-Fri minus KR holidays)."""
    year = today.year
    start = date(year, 1, 1)
    end_year = date(year, 12, 31)
    hol = kr_holidays_for_year(year)
    elapsed = business_days_in_range(start, today, hol)
    total = business_days_in_range(start, end_year, hol)
    return (elapsed / total) if total else 0.0


def business_day_ratio_for_month(today: date) -> float:
    """Elapsed business days within the current month (for partial-month weighting)."""
    year, month = today.year, today.month
    start = date(year, month, 1)
    last_day = calendar.monthrange(year, month)[1]
    end_month = date(year, month, last_day)
    hol = kr_holidays_for_year(year)
    elapsed = business_days_in_range(start, today, hol)
    total = business_days_in_range(start, end_month, hol)
    return (elapsed / total) if total else 0.0


def total_business_days_in_year(year: int) -> int:
    hol = kr_holidays_for_year(year)
    start = date(year, 1, 1)
    end_year = date(year, 12, 31)
    return business_days_in_range(start, end_year, hol)

# -----------------------------
# Seasonal (월별 가중) pace helpers
# -----------------------------


def monthly_share_series(df_monthly: pd.DataFrame, year: int, value_col: str) -> pd.Series:
    """Return 12-length monthly share for given year & metric. If total=0 or missing, returns zeros."""
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
    """Cumulative expected share up to 'today' using last year's monthly distribution with partial current month by biz-day ratio."""
    shares = monthly_share_series(
        df_monthly, last_year, value_col)  # index 1..12
    if shares.sum() == 0:
        return 0.0
    m = today.month
    # full months before current month
    full_share = shares.loc[1:m-1].sum() if m > 1 else 0.0
    # partial current month by business-day ratio
    part = shares.loc[m] * business_day_ratio_for_month(today)
    return float(full_share + part)


# -----------------------------
# Figures
# -----------------------------
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


def figure_weekly_today_based(df_daily: pd.DataFrame) -> go.Figure:
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
        text=[f"{v:,}" if v else "" for v in y_last], textposition='top center', textfont={'size': 11}
    ))
    fig.add_trace(go.Scatter(
        x=x_week, y=y_this, mode='lines+markers+text', name='이번 주',
        line=dict(width=3),
        customdata=this_dates_str,
        hovertemplate="%{customdata}<br>이번 주: %{y:,}부<extra></extra>",
        text=[f"{v:,}" if v else "" for v in y_this], textposition='top center', textfont={'size': 11}
    ))
    fig.update_layout(title=f'주간 발주량 비교 (기준일: {now.strftime("%Y-%m-%d")})',
                      xaxis_title='', yaxis_title='발주 부수', template='plotly_white', height=280,
                      margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation='h', x=1, xanchor='right', y=1.1))
    return fig


def figure_weekly_fixed_mon_fri(df_daily: pd.DataFrame, monday_str: str = None) -> go.Figure:
    if monday_str and monday_str != 'this_week':
        try:
            base_mon = datetime.strptime(monday_str, "%Y-%m-%d").date()
        except Exception:
            base_mon = monday(datetime.now(KST).date())
    else:
        base_mon = monday(datetime.now(KST).date())

    week_days = [base_mon + timedelta(days=i) for i in range(5)]  # Mon..Fri
    prev_week_days = [d - timedelta(days=7) for d in week_days]

    m = df_daily.set_index('date_only')[
        '총발주부수'].to_dict() if not df_daily.empty else {}
    y_this = [m.get(d, 0) for d in week_days]
    y_last = [m.get(d, 0) for d in prev_week_days]
    x_week = [WEEKDAY_KR[pd.Timestamp(d).weekday()] for d in week_days]
    this_dates_str = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in week_days]
    last_dates_str = [pd.Timestamp(d).strftime('%Y-%m-%d')
                      for d in prev_week_days]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_week, y=y_last, mode='lines+markers+text', name='지난 주',
        line=dict(width=2, dash='dot'),
        customdata=last_dates_str,
        hovertemplate="%{customdata}<br>지난 주: %{y:,}부<extra></extra>",
        text=[f"{v:,}" if v else "" for v in y_last], textposition='top center', textfont={'size': 11}
    ))
    fig.add_trace(go.Scatter(
        x=x_week, y=y_this, mode='lines+markers+text', name='이번 주',
        line=dict(width=3),
        customdata=this_dates_str,
        hovertemplate="%{customdata}<br>이번 주: %{y:,}부<extra></extra>",
        text=[f"{v:,}" if v else "" for v in y_this], textposition='top center', textfont={'size': 11}
    ))
    title_range = f"{week_days[0].strftime('%Y-%m-%d')} ~ {week_days[-1].strftime('%Y-%m-%d')}"
    fig.update_layout(title=f'주간 발주량 비교 (월~금 고정): {title_range}',
                      xaxis_title='', yaxis_title='발주 부수', template='plotly_white', height=300,
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
        hovertemplate="증감율: %{y:.1f}%<extra></extra>"
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

# -----------------------------
# KPI + Advanced progress
# -----------------------------


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
    """
    Returns dict with:
      - ytd_curr
      - ytd_ly
      - last_year_total
      - ratio_biz (영업일 경과율, 0~1)
      - progress_vs_biz (%)
      - seasonal_share_to_date (0~1)
      - progress_vs_seasonal (%)
    """
    today = datetime.now(KST).date()
    # YTD current
    ytd_curr = df_daily[(df_daily['연도'] == year) & (
        df_daily['date_only'] <= today)]['총발주부수'].sum()

    # last year YTD / total (from DAILY to be consistent)
    ly = year - 1
    same_date_ly = safe_same_date_last_year(today)
    ytd_ly = df_daily[(df_daily['연도'] == ly) & (
        df_daily['date_only'] <= same_date_ly)]['총발주부수'].sum()
    last_year_total = df_daily[df_daily['연도'] == ly]['총발주부수'].sum()

    # business-day based progress target
    ratio_biz = business_day_ratio_for_year(today)  # 0~1
    target_biz = last_year_total * ratio_biz if last_year_total else 0
    progress_vs_biz = (ytd_curr / target_biz * 100) if target_biz else None

    # seasonal (월별 가중) progress target using last year's monthly distribution
    seasonal_share = seasonal_cum_share_to_date(
        df_monthly, '발주량', ly, today)  # 0~1
    target_seasonal = last_year_total * seasonal_share if last_year_total else 0
    progress_vs_seasonal = (ytd_curr / target_seasonal *
                            100) if target_seasonal else None

    return {
        'ytd_curr': int(ytd_curr),
        'ytd_ly': int(ytd_ly),
        'last_year_total': int(last_year_total),
        'ratio_biz': ratio_biz,
        'progress_vs_biz': progress_vs_biz,
        'seasonal_share_to_date': seasonal_share,
        'progress_vs_seasonal': progress_vs_seasonal
    }


def badge(text, color="#2b8a3e"):
    return html.Span(text, style={
        'display': 'inline-block', 'padding': '4px 8px', 'borderRadius': '999px', 'background': color, 'color': 'white', 'fontSize': '0.8rem', 'fontWeight': '700'
    })


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
    # 1행: 총/평균/일수
    row1 = html.Div(style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap', 'marginBottom': '8px'}, children=[
        kpi_card(f"{year}년 총 발주량", f"{tot:,}", "Total Orders"),
        kpi_card(f"{year}년 일 평균 발주량", f"{avg:,}", "Avg / Working Day"),
        kpi_card(f"{year}년 총 발주 일수", f"{days:,}일", "Working Days Count"),
    ])
    # 2행: 흑백/컬러 (항상 아래 고정)
    row2 = html.Div(style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap', 'marginBottom': '6px'}, children=[
        kpi_card(f"{year}년 흑백 페이지 합계", f"{bw:,}", "BW Pages (Daily sum)"),
        kpi_card(f"{year}년 컬러 페이지 합계", f"{color:,}",
                 "Color Pages (Daily sum)"),
    ])

    # 3행: 고급 경과율 배지
    prog = compute_progress_advanced(df_daily, df_monthly, year)
    badges = []
    # YTD vs 작년 동기간
    if prog['ytd_ly']:
        ytd_vs_ly = prog['ytd_curr'] / prog['ytd_ly'] * 100
        badges.append(badge(f"YTD vs 작년 동기간: {ytd_vs_ly:.1f}%", "#0b7285"))
    # 영업일 경과율 기반
    if prog['progress_vs_biz'] is not None:
        color_biz = "#2b8a3e" if prog['progress_vs_biz'] >= 100 else "#d9480f"
        badges.append(
            badge(f"경과율(영업일) 대비 달성도: {prog['progress_vs_biz']:.1f}%", color_biz))
    # 월별 가중 pace 기반
    if prog['progress_vs_seasonal'] is not None:
        color_season = "#2b8a3e" if prog['progress_vs_seasonal'] >= 100 else "#d9480f"
        badges.append(
            badge(f"월별 가중 pace 대비 달성도: {prog['progress_vs_seasonal']:.1f}%", color_season))

    # 부가 배지: 영업일 경과율, 월별 가중 누적비중(설명용)
    badges.append(badge(f"영업일 경과율: {prog['ratio_biz']*100:.1f}%", "#6c757d"))
    if prog['seasonal_share_to_date'] is not None:
        badges.append(
            badge(f"월별 가중 누적비중: {prog['seasonal_share_to_date']*100:.1f}%", "#6c757d"))

    row3 = html.Div(style={'display': 'flex', 'gap': '8px', 'flexWrap': 'wrap',
                    'alignItems': 'center', 'margin': '4px 2px 0'}, children=badges)

    return html.Div(children=[row1, row2, row3])

# -----------------------------
# Forecast (연간 예측) 탭
# -----------------------------


def forecast_totals(df_daily: pd.DataFrame, df_monthly: pd.DataFrame, year: int):
    """Return dict of forecasts for total orders, bw pages, color pages and avg/day (business-day year)."""
    today = datetime.now(KST).date()
    ly = year - 1

    # YTD actuals
    ytd_orders = df_daily[(df_daily['연도'] == year) & (
        df_daily['date_only'] <= today)]['총발주부수'].sum()
    ytd_bw = df_daily[(df_daily['연도'] == year) & (
        df_daily['date_only'] <= today)]['흑백페이지'].sum()
    ytd_color = df_daily[(df_daily['연도'] == year) & (
        df_daily['date_only'] <= today)]['컬러페이지'].sum()

    # Ratios / shares
    ratio_biz_year = business_day_ratio_for_year(today) or 1e-9  # avoid zero
    share_orders = seasonal_cum_share_to_date(
        df_monthly, '발주량', ly, today) or None
    share_bw = seasonal_cum_share_to_date(
        df_monthly, '흑백출력량', ly, today) or None
    share_color = seasonal_cum_share_to_date(
        df_monthly, '컬러출력량', ly, today) or None

    # Forecast using seasonal share if available, else fall back to business-day ratio
    def forecast(ytd, share):
        if share and share > 0:
            return ytd / share
        return ytd / ratio_biz_year

    f_orders = forecast(ytd_orders, share_orders)
    f_bw = forecast(ytd_bw, share_bw)
    f_color = forecast(ytd_color, share_color)

    # Avg per business day (year)
    total_biz_days = total_business_days_in_year(year)
    avg_per_bd = (f_orders / total_biz_days) if total_biz_days else 0.0

    return {
        'forecast_orders': int(round(f_orders)),
        'forecast_bw': int(round(f_bw)),
        'forecast_color': int(round(f_color)),
        'forecast_avg_per_bd': round(avg_per_bd, 2),
        'method_orders': '월별가중' if (share_orders and share_orders > 0) else '영업일pace',
        'method_bw': '월별가중' if (share_bw and share_bw > 0) else '영업일pace',
        'method_color': '월별가중' if (share_color and share_color > 0) else '영업일pace',
    }


def forecast_cards_layout(df_daily: pd.DataFrame, df_monthly: pd.DataFrame, year: int):
    fc = forecast_totals(df_daily, df_monthly, year)

    def small_note(text):
        return html.Div(text, style={'fontSize': '0.8rem', 'color': '#8a8a8a', 'marginTop': '4px'})
    return html.Div(style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap'}, children=[
        html.Div(style={'flex': '1 1 260px', 'background': 'white', 'borderRadius': '14px', 'padding': '14px 16px', 'boxShadow': '0 6px 18px rgba(0,0,0,0.08)', 'minWidth': '240px'}, children=[
            html.Div("올해 예상 발주량", style={
                     'fontSize': '0.95rem', 'color': '#666', 'marginBottom': '6px', 'fontWeight': '600'}),
            html.Div(f"{fc['forecast_orders']:,}", style={
                     'fontSize': '1.8rem', 'fontWeight': '800'}),
            small_note(f"방법: {fc['method_orders']} · 연말 예상 총합")
        ]),
        html.Div(style={'flex': '1 1 260px', 'background': 'white', 'borderRadius': '14px', 'padding': '14px 16px', 'boxShadow': '0 6px 18px rgba(0,0,0,0.08)', 'minWidth': '240px'}, children=[
            html.Div("올해 예상 일평균 발주량(영업일)", style={
                     'fontSize': '0.95rem', 'color': '#666', 'marginBottom': '6px', 'fontWeight': '600'}),
            html.Div(f"{fc['forecast_avg_per_bd']:,}", style={
                     'fontSize': '1.8rem', 'fontWeight': '800'}),
            small_note("총 영업일 수로 나눈 예상 평균")
        ]),
        html.Div(style={'flex': '1 1 260px', 'background': 'white', 'borderRadius': '14px', 'padding': '14px 16px', 'boxShadow': '0 6px 18px rgba(0,0,0,0.08)', 'minWidth': '240px'}, children=[
            html.Div("올해 예상 흑백 페이지", style={
                     'fontSize': '0.95rem', 'color': '#666', 'marginBottom': '6px', 'fontWeight': '600'}),
            html.Div(f"{fc['forecast_bw']:,}", style={
                     'fontSize': '1.8rem', 'fontWeight': '800'}),
            small_note(f"방법: {fc['method_bw']}")
        ]),
        html.Div(style={'flex': '1 1 260px', 'background': 'white', 'borderRadius': '14px', 'padding': '14px 16px', 'boxShadow': '0 6px 18px rgba(0,0,0,0.08)', 'minWidth': '240px'}, children=[
            html.Div("올해 예상 컬러 페이지", style={
                     'fontSize': '0.95rem', 'color': '#666', 'marginBottom': '6px', 'fontWeight': '600'}),
            html.Div(f"{fc['forecast_color']:,}", style={
                     'fontSize': '1.8rem', 'fontWeight': '800'}),
            small_note(f"방법: {fc['method_color']}")
        ]),
    ])


# -----------------------------
# App
# -----------------------------
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
    DF_DAILY['연도'].unique()) if int(y) <= CURRENT_YEAR] if not DF_DAILY.empty else []

# Charts
fig_week_today = figure_weekly_today_based(
    DF_DAILY) if not DF_DAILY.empty else go.Figure()
fig_months = figure_months_1to12(
    DF_MONTHLY, start_year=2022, current_year=CURRENT_YEAR) if not DF_MONTHLY.empty else go.Figure()
metric_figs = build_metric_figs(DF_MONTHLY, baseline_year=CURRENT_YEAR) if not DF_MONTHLY.empty else {
    'avg_per_day': go.Figure(), 'monthly_total': go.Figure(), 'bw_pages': go.Figure(), 'color_pages': go.Figure()}

week_options = week_options_from_df(DF_DAILY)

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
        html.Div([
            html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}, children=[
                html.H3("주간 발주량 비교 (오늘 기준 5영업일 + 지난주 같은 요일)",
                        style={'marginBottom': '4px', 'fontSize': '1.05rem'}),
            ]),
            dcc.Graph(id='weekly-chart-today', figure=fig_week_today,
                      style={'height': '300px'})
        ], style=CARD_STYLE),
    ]),

    html.Details(open=False, children=[
        html.Summary("월~금 고정 주간 비교 (클릭하여 열기)"),
        html.Div(style={'display': 'flex', 'gap': '8px', 'alignItems': 'center', 'margin': '8px 0'}, children=[
            html.Span("주차 선택:", style={'fontWeight': '600'}),
            dcc.Dropdown(id='week-select-fixed', options=week_options,
                         value='this_week', clearable=False, style={'width': '300px'}),
        ]),
        dcc.Graph(id='weekly-chart-fixed', style={'height': '320px'})
    ], style={**CARD_STYLE, 'marginBottom': '12px'}),

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
            dcc.Tab(label="연간 예측", value="forecast"),
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
        kpi_layout = build_kpi_layout(DF_DAILY, DF_MONTHLY, selected_year)

        # 과거 연도 간단 표 (정렬 스타일 적용)
        if DF_DAILY.empty:
            prev_tbl = html.Div()
        else:
            years = sorted(DF_DAILY['연도'].unique())

            # 정렬/가독성 스타일
            td_year_style = {
                'textAlign': 'left', 'padding': '6px 8px', 'borderBottom': '1px solid #f1f3f5'}
            td_num_style = {
                'textAlign': 'right', 'padding': '6px 8px', 'borderBottom': '1px solid #f1f3f5',
                'fontVariantNumeric': 'tabular-nums', 'whiteSpace': 'nowrap'
            }
            th_year_style = {'textAlign': 'left', 'padding': '6px 8px',
                             'borderBottom': '2px solid #dee2e6', 'color': '#495057'}
            th_num_style = {
                'textAlign': 'right', 'padding': '6px 8px', 'borderBottom': '2px solid #dee2e6', 'color': '#495057',
                'fontVariantNumeric': 'tabular-nums', 'whiteSpace': 'nowrap'
            }

            rows = []
            for y in years:
                if y == selected_year:
                    continue
                tot, avg, days, _, _ = compute_kpis(DF_DAILY, y)
                rows.append(html.Tr([
                    html.Td(f"{y}년",   style=td_year_style),
                    html.Td(f"{tot:,}", style=td_num_style),
                    html.Td(f"{avg:,}", style=td_num_style),
                    html.Td(f"{days:,}", style=td_num_style),
                ]))

            header = html.Tr([
                html.Th("연도",            style=th_year_style),
                html.Th("총 발주량",       style=th_num_style),
                html.Th("일 평균 발주량",  style=th_num_style),
                html.Th("총 발주 일수",    style=th_num_style),
            ])

            prev_tbl = html.Table(
                [header] + rows,
                style={'width': '100%', 'borderCollapse': 'collapse',
                       'tableLayout': 'fixed'},
                className="kpi-table"
            )

        stamp = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")
        return kpi_layout, prev_tbl, f"업데이트: {stamp}"
    except Exception as e:
        return html.Div("KPI 계산 오류"), html.Div(f"오류: {e}"), ""


@callback(
    Output("metric-tab-content", "children"),
    Input("metric-tabs", "value"),
    Input("year-select", "value")
)
def switch_metric_tab(tab_value, selected_year):
    figs = {
        "avg": metric_figs['avg_per_day'],
        "total": metric_figs['monthly_total'],
        "bw": metric_figs['bw_pages'],
        "color": metric_figs['color_pages'],
    }
    if tab_value == "forecast":
        layout = forecast_cards_layout(DF_DAILY, DF_MONTHLY, selected_year)
        return html.Div(style={'padding': '10px'}, children=[layout])
    fig = figs.get(tab_value, go.Figure())
    fig.update_layout(height=460)
    return dcc.Graph(figure=fig, style={'height': '480px'})


@callback(
    Output('weekly-chart-fixed', 'figure'),
    Input('week-select-fixed', 'value')
)
def update_week_fixed(monday_str):
    return figure_weekly_fixed_mon_fri(DF_DAILY, monday_str)


if __name__ == '__main__':
    host = os.getenv('DASH_HOST', '127.0.0.1')
    port = int(os.getenv('DASH_PORT', '8090'))
    print(f"브라우저에서 http://{host}:{port} 으로 접속하세요.")
    app.run(debug=True, host=host, port=port)

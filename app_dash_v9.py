import os
import re
import calendar
from datetime import datetime, timedelta, date

import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials

import dash
from dash import dcc, html, Input, Output, State, callback
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go

from dotenv import load_dotenv
import pytz

load_dotenv()
KST = pytz.timezone('Asia/Seoul')

# ===== 설정 =====
AUTO_REFRESH_MIN = int(os.getenv("AUTO_REFRESH_MIN", "15"))
MOVAVG_MONTHS = int(os.getenv("MOVAVG_MONTHS", "3"))           # 이동평균 개월 수
# (참고 하한, MA 5지표에는 미사용)
MA_MIN_PROGRESS = float(os.getenv("MA_MIN_PROGRESS", "0.10"))

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

    url = os.getenv("SPREADSHEET_URL", "").strip()
    if not url:
        raise EnvironmentError("SPREADSHEET_URL이 .env/환경변수에 없습니다.")
    return client.open_by_url(url)


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

# -----------------------------------------------------------------------------
# 일/월 시트 정제
# -----------------------------------------------------------------------------


def cleanse_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=['날짜', 'date_only', '연도', '월',
                                     '총발주종수', '총발주부수', '흑백페이지', '컬러페이지',
                                     '예상제본시간', '최종출고', '출고부수'])
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

    # 공란일 제거(존재하는 주요열 모두 공란일 때만 제거)
    candidates = [col_cnt, col_total, col_bw, col_color]
    existing = [c for c in candidates if c and c in d.columns]
    if existing:
        blanks = pd.Series(True, index=d.index)
        for c in existing:
            blanks = blanks & d[c].astype(str).str.strip().eq('').fillna(True)
        d = d.loc[~blanks].copy()

    def to_num(series):
        return pd.to_numeric(series.astype(str).str.replace(',', '', regex=False), errors='coerce').fillna(0).astype(int)

    d['총발주종수'] = to_num(d[col_cnt]) if (
        col_cnt and col_cnt in d.columns) else 0
    d['총발주부수'] = to_num(d[col_total]) if (
        col_total and col_total in d.columns) else 0
    d['흑백페이지'] = to_num(d[col_bw]) if (col_bw and col_bw in d.columns) else 0
    d['컬러페이지'] = to_num(d[col_color]) if (
        col_color and col_color in d.columns) else 0
    d['출고부수'] = to_num(d[col_ships]) if (
        col_ships and col_ships in d.columns) else 0

    def as_text(colname):
        if colname and colname in d.columns:
            s = d[colname].astype(str).str.strip()
            s = s.where(~s.eq(''), '-')
            return s
        return '-'

    d['예상제본시간'] = as_text(col_bind)
    d['최종출고'] = as_text(col_shipt)

    d['date_only'] = d['날짜'].dt.date
    d['연도'] = d['날짜'].dt.year
    d['월'] = d['날짜'].dt.month

    return d[['날짜', 'date_only', '연도', '월',
              '총발주종수', '총발주부수', '흑백페이지', '컬러페이지',
              '예상제본시간', '최종출고', '출고부수']]


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
        import holidays  # type: ignore
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


def business_day_ratio_for_month(today: date) -> float:
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


def figure_weekly_today_based(df_daily: pd.DataFrame, value_col: str = '총발주부수') -> go.Figure:
    now = datetime.now(KST)
    this_week_dates = last_5_business_days_upto_today(now)
    last_week_dates = [d - timedelta(days=7) for d in this_week_dates]

    metric_map = {'총발주부수': '발주량', '흑백페이지': '흑백 페이지', '컬러페이지': '컬러 페이지'}
    unit = '부' if value_col == '총발주부수' else '페이지'
    metric_name = metric_map.get(value_col, '값')

    m = df_daily.set_index('date_only')[value_col].to_dict(
    ) if not df_daily.empty and value_col in df_daily.columns else {}
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
        line=dict(width=2, dash='dot'), customdata=last_dates_str,
        hovertemplate="%{customdata}<br>지난 주: %{y:,}"+unit+"<extra></extra>",
        text=[f"{v:,}" if v else "" for v in y_last], textposition='top center', textfont={'size': 11}
    ))
    fig.add_trace(go.Scatter(
        x=x_week, y=y_this, mode='lines+markers+text', name='이번 주',
        line=dict(width=3), customdata=this_dates_str,
        hovertemplate="%{customdata}<br>이번 주: %{y:,}"+unit+"<extra></extra>",
        text=[f"{v:,}" if v else "" for v in y_this], textposition='top center', textfont={'size': 11}
    ))
    fig.update_layout(title=f'주간 {metric_name} 비교 (기준일: {now.strftime("%Y-%m-%d")})',
                      xaxis_title='', yaxis_title=f'{metric_name} ({unit})', template='plotly_white', height=280,
                      margin=dict(l=20, r=20, t=40, b=20),
                      legend=dict(orientation='h', x=1, xanchor='right', y=1.1))
    return fig


def figure_weekly_fixed_mon_fri(df_daily: pd.DataFrame, monday_str: str = None, value_col: str = '총발주부수') -> go.Figure:
    if monday_str and monday_str != 'this_week':
        try:
            base_mon = datetime.strptime(monday_str, "%Y-%m-%d").date()
        except Exception:
            base_mon = monday(datetime.now(KST).date())
    else:
        base_mon = monday(datetime.now(KST).date())

    week_days = [base_mon + timedelta(days=i) for i in range(5)]
    prev_week_days = [d - timedelta(days=7) for d in week_days]

    metric_map = {'총발주부수': '발주량', '흑백페이지': '흑백 페이지', '컬러페이지': '컬러 페이지'}
    unit = '부' if value_col == '총발주부수' else '페이지'
    metric_name = metric_map.get(value_col, '값')

    m = df_daily.set_index('date_only')[value_col].to_dict(
    ) if not df_daily.empty and value_col in df_daily.columns else {}
    y_this = [m.get(d, 0) for d in week_days]
    y_last = [m.get(d, 0) for d in prev_week_days]
    x_week = [WEEKDAY_KR[pd.Timestamp(d).weekday()] for d in week_days]
    this_dates_str = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in week_days]
    last_dates_str = [pd.Timestamp(d).strftime('%Y-%m-%d')
                      for d in prev_week_days]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_week, y=y_last, mode='lines+markers+text', name='지난 주',
        line=dict(width=2, dash='dot'), customdata=last_dates_str,
        hovertemplate="%{customdata}<br>지난 주: %{y:,}"+unit+"<extra></extra>",
        text=[f"{v:,}" if v else "" for v in y_last], textposition='top center', textfont={'size': 11}
    ))
    fig.add_trace(go.Scatter(
        x=x_week, y=y_this, mode='lines+markers+text', name='이번 주',
        line=dict(width=3), customdata=this_dates_str,
        hovertemplate="%{customdata}<br>이번 주: %{y:,}"+unit+"<extra></extra>",
        text=[f"{v:,}" if v else "" for v in y_this], textposition='top center', textfont={'size': 11}
    ))
    title_range = f"{week_days[0].strftime('%Y-%m-%d')} ~ {week_days[-1].strftime('%Y-%m-%d')}"
    fig.update_layout(title=f'주간 {metric_name} 비교 (월~금 고정): {title_range}',
                      xaxis_title='', yaxis_title=f'{metric_name} ({unit})', template='plotly_white', height=300,
                      margin=dict(l=20, r=20, t=40, b=20),
                      legend=dict(orientation='h', x=1, xanchor='right', y=1.1))
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
                      margin=dict(l=20, r=20, t=40, b=20),
                      legend=dict(orientation='h', x=1, xanchor='right', y=1.1))
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
    today = datetime.now(KST).date()
    ly = year - 1

    ytd_curr = df_daily[(df_daily['연도'] == year) & (
        df_daily['date_only'] <= today)]['총발주부수'].sum()
    same_date_ly = safe_same_date_last_year(today)
    ytd_ly = df_daily[(df_daily['연도'] == ly) & (
        df_daily['date_only'] <= same_date_ly)]['총발주부수'].sum()
    last_year_total = df_daily[df_daily['연도'] == ly]['총발주부수'].sum()

    hol = kr_holidays_for_year(year)
    elapsed_biz_days = business_days_in_range(date(year, 1, 1), today, hol)
    total_biz_days = business_days_in_range(
        date(year, 1, 1), date(year, 12, 31), hol)
    ratio_biz = (elapsed_biz_days / total_biz_days) if total_biz_days else 0.0

    target_biz = last_year_total * ratio_biz if last_year_total else 0
    progress_vs_biz = (ytd_curr / target_biz * 100) if target_biz else None

    seasonal_share = seasonal_cum_share_to_date(df_monthly, '발주량', ly, today)
    target_seasonal = last_year_total * seasonal_share if last_year_total else 0
    progress_vs_seasonal = (ytd_curr / target_seasonal *
                            100) if target_seasonal else None

    # 참고용 MA (기존 막대그래프 보조치)
    m = today.month
    ma_n = max(1, MOVAVG_MONTHS)
    cur_y = df_monthly[(df_monthly['연도'] == year) & (df_monthly['월번호'] <= m)]
    cur_months = cur_y[['월번호', '발주량']].sort_values('월번호')
    vals = cur_months['발주량'].tolist()
    need = ma_n - len(vals)
    if need > 0:
        prev = df_monthly[(df_monthly['연도'] == ly)].sort_values('월번호')
        take = prev['발주량'].tolist()[-need:] if not prev.empty else []
        vals = take + vals
    if len(vals) == 0:
        ma_recent = 0.0
    else:
        ma_recent = float(pd.Series(vals[-ma_n:]).mean())
    share_safe = max(seasonal_share or 0.0, MA_MIN_PROGRESS)
    forecast_ma = int(round(ma_recent * 12 / share_safe)
                      ) if ma_recent > 0 else None

    return {
        'ytd_curr': int(ytd_curr),
        'ytd_ly': int(ytd_ly),
        'last_year_total': int(last_year_total),
        'ratio_biz': ratio_biz,
        'progress_vs_biz': progress_vs_biz,
        'seasonal_share_to_date': seasonal_share,
        'progress_vs_seasonal': progress_vs_seasonal,
        'elapsed_biz_days': elapsed_biz_days,
        'total_biz_days': total_biz_days,
        'today': today,
        'same_date_ly': same_date_ly,
        'ma_recent': ma_recent,
        'forecast_ma': forecast_ma,
        'ma_n': ma_n,
        'share_safe': share_safe
    }

# ===== 이동평균 기반 연간 예측 5종 계산 (견고 버전) =====


def _ma_for_col(df_monthly: pd.DataFrame, year: int, col: str, ma_n: int) -> float:
    """
    최근 ma_n개월 이동평균(현재 연도에 값이 없거나 0이라도, 직전 연도까지 거슬러서
    '값이 있는' 월 기준으로 계산). 0 또는 비어있으면 0.0 반환.
    """
    if df_monthly.empty or col not in df_monthly.columns:
        return 0.0
    sub = df_monthly[df_monthly['연도'] <= year].sort_values('월DT')
    if sub.empty:
        return 0.0
    # 값이 있는 월만 사용(0은 미기입/미발생으로 간주)
    sub = sub[sub[col].astype(float) > 0]
    if sub.empty:
        return 0.0
    return float(sub[col].astype(float).tail(ma_n).mean())


def compute_ma_forecasts(df_monthly: pd.DataFrame, year: int, ma_n: int) -> dict:
    """
    이동평균(MA) 기반 올해 예상치 5종:
    - 총 발주량/총 발주 일수/흑백/컬러 페이지 합계: 월간 MA × 12
    - 일 평균 발주량: 월별 '일평균발주량'의 MA(연간화하지 않음)
    """
    if df_monthly.empty:
        return {k: None for k in ['총 발주량', '일 평균 발주량', '총 발주 일수', '흑백 페이지 합계', '컬러 페이지 합계']}

    ma_total = _ma_for_col(df_monthly, year, '발주량',       ma_n)
    ma_days = _ma_for_col(df_monthly, year, '발주일수',     ma_n)
    ma_bw = _ma_for_col(df_monthly, year, '흑백출력량',   ma_n)
    ma_color = _ma_for_col(df_monthly, year, '컬러출력량',   ma_n)
    ma_dailyavg = _ma_for_col(df_monthly, year, '일평균발주량', ma_n)

    return {
        '총 발주량':       int(round(ma_total * 12)) if ma_total > 0 else 0,
        '일 평균 발주량':   round(ma_dailyavg, 1) if ma_dailyavg > 0 else 0.0,
        '총 발주 일수':     int(round(ma_days * 12)) if ma_days > 0 else 0,
        '흑백 페이지 합계': int(round(ma_bw * 12)) if ma_bw > 0 else 0,
        '컬러 페이지 합계': int(round(ma_color * 12)) if ma_color > 0 else 0
    }

# -----------------------------------------------------------------------------
# KPI/배지/오늘패널
# -----------------------------------------------------------------------------


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
    # ===== 요청 포맷으로 4개 배지 구성 =====
    badges = []
    # 1) YTD vs 작년 동기간
    if prog['ytd_ly'] > 0:
        ytd_vs_ly = prog['ytd_curr'] / prog['ytd_ly'] * 100
        badges.append(badge(f"YTD vs 작년 동기간: {ytd_vs_ly:.1f}%", "#0b7285"))

    # 2) 경과율(영업일) 대비
    if prog['progress_vs_biz'] is not None:
        badges.append(badge(f"경과율(영업일) 대비: {prog['progress_vs_biz']:.1f}%",
                      "#2b8a3e" if prog['progress_vs_biz'] >= 100 else "#d9480f"))

    # 3) 월별 가중치 대비
    if prog['progress_vs_seasonal'] is not None:
        badges.append(badge(f"월별 가중치 대비: {prog['progress_vs_seasonal']:.1f}%",
                      "#2b8a3e" if prog['progress_vs_seasonal'] >= 100 else "#d9480f"))

    # 4) 영업일 경과율
    badges.append(badge(f"영업일 경과율: {prog['ratio_biz']*100:.1f}%", "#5f3dc4"))

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

            def fmt_int(x):
                try:
                    return f"{int(x):,}"
                except Exception:
                    return "0"
            total_kinds = fmt_int(r.get('총발주종수', 0))
            total_qty = fmt_int(r.get('총발주부수', 0))
            bw_pages = fmt_int(r.get('흑백페이지', 0))
            color_pages = fmt_int(r.get('컬러페이지', 0))
            bind_time = (str(r.get('예상제본시간', '-')) or '-')
            last_ship = (str(r.get('최종출고', '-')) or '-')
            ship_qty = fmt_int(r.get('출고부수', 0))
            grid = html.Div(style={
                'display': 'grid', 'gridTemplateColumns': '1fr 1fr',
                'rowGap': '6px', 'columnGap': '8px'
            }, children=[
                html.Div("총 발주 종수", style={'color': '#666'}), html.Div(
                    total_kinds, style={'textAlign': 'right', 'fontWeight': '700'}),
                html.Div("총 발주 부수", style={'color': '#666'}), html.Div(
                    total_qty,   style={'textAlign': 'right', 'fontWeight': '700'}),
                html.Div("흑백 페이지",  style={'color': '#666'}), html.Div(
                    bw_pages,    style={'textAlign': 'right', 'fontWeight': '700'}),
                html.Div("컬러 페이지",  style={'color': '#666'}), html.Div(
                    color_pages, style={'textAlign': 'right', 'fontWeight': '700'}),
                html.Div("예상 제본 시간", style={'color': '#666'}), html.Div(
                    bind_time,  style={'textAlign': 'right', 'fontWeight': '700'}),
                html.Div("최종 출고",    style={'color': '#666'}), html.Div(
                    last_ship,   style={'textAlign': 'right', 'fontWeight': '700'}),
                html.Div("출고 부수",    style={'color': '#666'}), html.Div(
                    ship_qty,    style={'textAlign': 'right', 'fontWeight': '700'}),
            ])
            body = grid

    header = html.Div([
        html.Div("오늘 현황", style={'fontWeight': '800', 'fontSize': '1.05rem'}),
        html.Div(datetime.now(KST).strftime("%Y-%m-%d (%a) %H:%M"),
                 style={'color': '#888', 'fontSize': '0.8rem', 'marginTop': '2px'})
    ])

    return html.Div(style={
        'position': 'fixed', 'top': '80px', 'right': '16px',
        'width': 'min(94vw, 310px)', 'zIndex': '999',
        'background': 'rgba(255,255,255,0.98)', 'backdropFilter': 'blur(2px)',
        'border': '1px solid #edf2f7', 'borderRadius': '14px', 'padding': '12px 14px',
        'boxShadow': '0 10px 22px rgba(0,0,0,0.12)'
    }, children=[header, html.Hr(style={'margin': '8px 0', 'borderColor': '#f1f3f5'}), body])

# (호환용) 만약 옛 코드가 make_today_panel을 호출하면 build_today_panel 사용


def make_today_panel(df_daily: pd.DataFrame):
    return build_today_panel(df_daily)

# -----------------------------------------------------------------------------
# 예측 섹션 (MA 기반 5개 지표 숫자 요약)
# -----------------------------------------------------------------------------


def forecast_cards_layout(df_daily, df_monthly, year: int):
    """이동평균(MA) 기반 올해 예상 5개 지표를 숫자로 표시."""
    ma_n = max(1, MOVAVG_MONTHS)
    fx = compute_ma_forecasts(df_monthly, year, ma_n)

    def kv_row(k, v, hint=""):
        val = f"{v:,}" if isinstance(v, int) else f"{v:,.1f}"
        sub = html.Span(
            hint, style={'color': '#8a8a8a', 'fontSize': '0.8rem'}) if hint else None
        return html.Div(style={
            'display': 'grid', 'gridTemplateColumns': '1fr auto', 'alignItems': 'baseline',
            'padding': '8px 0', 'borderBottom': '1px solid #f1f3f5'
        }, children=[
            html.Div([html.Strong(k), html.Span(" "), sub]),
            html.Div(val, style={'fontWeight': '800', 'fontSize': '1.1rem'})
        ])

    header = html.Div([
        html.Div(f"{year}년 예상치 (이동평균 {ma_n}개월 기반)",
                 style={'fontWeight': '800', 'fontSize': '1.05rem', 'marginBottom': '2px'}),
        html.Div("월별 지표의 최근 N개월 평균을 연간으로 환산합니다. "
                 "‘일 평균 발주량’은 월별 일평균의 이동평균(연간화하지 않음).",
                 style={'color': '#888', 'fontSize': '0.85rem'})
    ], style={'marginBottom': '8px'})

    grid = html.Div(style={
        'background': 'white', 'borderRadius': '12px', 'padding': '12px',
        'boxShadow': '0 4px 14px rgba(0,0,0,0.08)'
    }, children=[
        kv_row("총 발주량",       fx['총 발주량'],       "MA×12"),
        kv_row("일 평균 발주량",   fx['일 평균 발주량'],   f"MA({ma_n}개월)"),
        kv_row("총 발주 일수",     fx['총 발주 일수'],     "MA×12"),
        kv_row("흑백 페이지 합계", fx['흑백 페이지 합계'], "MA×12"),
        kv_row("컬러 페이지 합계", fx['컬러 페이지 합계'], "MA×12"),
    ])

    return html.Div(children=[header, grid])


# -----------------------------------------------------------------------------
# 앱 (지연 로딩 레이아웃)
# -----------------------------------------------------------------------------
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
CURRENT_YEAR = datetime.now(KST).year

app.layout = html.Div(style={'maxWidth': '1100px', 'margin': '0 auto', 'padding': '16px',
                             'fontFamily': 'Noto Sans KR, Malgun Gothic, Arial'}, children=[
    dcc.Interval(id='init', interval=250, n_intervals=0, max_intervals=1),
    dcc.Interval(id='today-refresh', interval=120*1000, n_intervals=0),
    dcc.Interval(id='data-auto-refresh',
                 interval=AUTO_REFRESH_MIN*60*1000, n_intervals=0),

    dcc.Store(id='data-version', storage_type='memory', data=0),
    dcc.Store(id='today-visible', storage_type='local', data=True),
    dcc.Store(id='memo-visible', storage_type='local', data=False),
    dcc.Store(id='memo-storage', storage_type='local', data=""),

    html.H1("발주량 분석 대시보드", style={
            'textAlign': 'center', 'marginBottom': '6px'}),
    html.P("구글 시트 데이터 기반 · 2021~현재", style={
           'textAlign': 'center', 'marginBottom': '14px', 'color': '#666'}),

    html.Div(style={'display': 'flex', 'gap': '10px', 'justifyContent': 'center', 'alignItems': 'center',
                    'marginBottom': '8px', 'flexWrap': 'wrap'}, children=[
        html.Span("KPI 연도 선택:", style={'fontWeight': '600'}),
        dcc.Dropdown(id='year-select',
                     options=[
                         {'label': f'{CURRENT_YEAR}년', 'value': CURRENT_YEAR}],
                     value=CURRENT_YEAR, clearable=False, style={'width': '220px'}),
        html.Button("데이터 새로고침", id="manual-refresh", n_clicks=0, title="구글 시트에서 다시 불러오기",
                    style={'borderRadius': '999px', 'padding': '8px 12px', 'fontWeight': '700',
                           'border': '1px solid #e2e8f0', 'background': '#2563eb',
                           'color': 'white', 'cursor': 'pointer'}),
        html.Div(id='kpi-refresh-status',
                 style={'marginLeft': '12px', 'color': '#888'})
    ]),

    html.Div(id='kpi-cards'),

    html.Details(open=False, children=[
        html.Summary("지난 연도 KPI 펼치기 / 접기"),
        html.Div(id='prev-years-kpi', style={'marginTop': '8px'})
    ], style={'background': 'white', 'borderRadius': '12px', 'padding': '14px',
              'boxShadow': '0 4px 14px rgba(0,0,0,0.08)', 'marginBottom': '16px'}),

    html.Div(style={'background': 'white', 'borderRadius': '12px', 'padding': '14px',
                    'boxShadow': '0 4px 14px rgba(0,0,0,0.08)', 'marginBottom': '12px'}, children=[
        html.H3("주간 비교 (오늘 기준 5영업일 vs 지난주 동요일)", style={
                'marginBottom': '8px', 'fontSize': '1.05rem'}),
        dcc.Tabs(id="weekly-tabs-today", value="총발주부수", children=[
            dcc.Tab(label="발주량", value="총발주부수"),
            dcc.Tab(label="흑백 페이지", value="흑백페이지"),
            dcc.Tab(label="컬러 페이지", value="컬러페이지"),
        ]),
        dcc.Graph(id='weekly-chart-today', figure=go.Figure(),
                  style={'height': '300px'}),
    ]),

    html.Details(open=False, children=[
        html.Summary("월~금 고정 주간 비교 (클릭하여 열기)"),
        html.Div(style={'display': 'flex', 'gap': '8px', 'alignItems': 'center', 'margin': '8px 0'}, children=[
            html.Span("주차 선택:", style={'fontWeight': '600'}),
            dcc.Dropdown(id='week-select-fixed', options=[{'label': '오늘 기준 (최근 5영업일)', 'value': 'this_week'}],
                         value='this_week', clearable=False, style={'width': '300px'}),
        ]),
        dcc.Tabs(id="weekly-tabs-fixed", value="총발주부수", children=[
            dcc.Tab(label="발주량", value="총발주부수"),
            dcc.Tab(label="흑백 페이지", value="흑백페이지"),
            dcc.Tab(label="컬러 페이지", value="컬러페이지"),
        ]),
        dcc.Graph(id='weekly-chart-fixed', style={'height': '320px'})
    ], style={'background': 'white', 'borderRadius': '12px', 'padding': '14px',
              'boxShadow': '0 4px 14px rgba(0,0,0,0.08)', 'marginBottom': '12px'}),

    html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr', 'gap': '12px', 'marginBottom': '12px'}, children=[
        html.Div([html.H3("월별 발주량 (1~12월, 2022~현재)", style={'marginBottom': '4px', 'fontSize': '1.05rem'}),
                  dcc.Graph(id='months-1to12-chart', figure=go.Figure(), style={'height': '320px'})],
                 style={'background': 'white', 'borderRadius': '12px', 'padding': '14px',
                        'boxShadow': '0 4px 14px rgba(0,0,0,0.08)'}),
    ]),

    html.Div(style={'background': 'white', 'borderRadius': '12px', 'padding': '0px',
                    'boxShadow': '0 4px 14px rgba(0,0,0,0.08)'}, children=[
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

    # 메모장 팝업
    html.Div(id='memo-popup', style={
        'position': 'fixed', 'left': '16px', 'top': '80px',
        'width': 'min(92vw, 360px)', 'zIndex': '1000',
        'display': 'none',
        'background': 'rgba(255,255,255,0.98)', 'backdropFilter': 'blur(2px)',
        'border': '1px solid #edf2f7', 'borderRadius': '14px', 'padding': '10px 12px',
        'boxShadow': '0 10px 22px rgba(0,0,0,0.12)', 'maxHeight': '70vh', 'overflowY': 'auto'
    }, children=[
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '6px'}, children=[
            html.Div("메모장", style={'fontWeight': '800', 'fontSize': '1.0rem'}),
            html.Div([
                html.Button("저장", id="memo-save-btn", n_clicks=0,
                            style={'marginRight': '6px'}),
                html.Button("지우기", id="memo-clear-btn", n_clicks=0,
                            style={'marginRight': '6px'}),
                html.Button("닫기", id="memo-close-btn", n_clicks=0),
            ])
        ]),
        dcc.Textarea(
            id='memo-text', style={'width': '100%', 'height': '30vh', 'resize': 'vertical'}, spellCheck=False),
        html.Div(style={'marginTop': '6px', 'textAlign': 'right',
                 'color': '#888', 'fontSize': '0.8rem'}, id='memo-status')
    ]),

    html.Button("오늘 현황", id="fab-today-toggle", n_clicks=0, title="오른쪽 패널 열기/닫기", style={
        'position': 'fixed', 'right': '16px', 'bottom': '16px',
        'borderRadius': '999px', 'padding': '12px 16px', 'fontWeight': '800',
        'background': '#2563eb', 'color': 'white', 'border': 'none',
        'boxShadow': '0 8px 18px rgba(0,0,0,0.18)', 'zIndex': 1100, 'cursor': 'pointer'
    }),
    html.Button("메모장", id="fab-memo-toggle", n_clicks=0, title="메모장 열기/닫기", style={
        'position': 'fixed', 'left': '16px', 'bottom': '16px',
        'borderRadius': '999px', 'padding': '12px 16px', 'fontWeight': '800',
        'background': '#059669', 'color': 'white', 'border': 'none',
        'boxShadow': '0 8px 18px rgba(0,0,0,0.18)', 'zIndex': 1100, 'cursor': 'pointer'
    }),
])

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
        DATA["daily"] = pd.DataFrame()
        DATA["monthly"] = pd.DataFrame()
        DATA["years_options"] = [
            {'label': f'{CURRENT_YEAR}년', 'value': CURRENT_YEAR}]
        DATA["week_options"] = [
            {'label': '오늘 기준 (최근 5영업일)', 'value': 'this_week'}]
        DATA["loaded"] = True  # 실패해도 앱은 동작


@callback(
    Output('data-version', 'data'),
    Input('data-auto-refresh', 'n_intervals'),
    Input('manual-refresh', 'n_clicks'),
    State('data-version', 'data'),
    prevent_initial_call=False
)
def auto_or_manual_reload(_n, _clicks, ver):
    try:
        DATA["loaded"] = False
        ensure_data_loaded()
        return (ver or 0) + 1
    except Exception:
        raise PreventUpdate

# -----------------------------------------------------------------------------
# 콜백
# -----------------------------------------------------------------------------


@callback(
    Output('year-select', 'options'),
    Output('year-select', 'value'),
    Input('init', 'n_intervals'),
    Input('data-version', 'data'),
    prevent_initial_call=False
)
def init_year_options(_, _ver):
    ensure_data_loaded()
    opts = DATA["years_options"]
    latest = max([o['value'] for o in opts]) if opts else CURRENT_YEAR
    return opts, latest


@callback(
    Output('kpi-cards', 'children'),
    Output('prev-years-kpi', 'children'),
    Output('kpi-refresh-status', 'children'),
    Input('year-select', 'value'),
    Input('data-version', 'data')
)
def update_kpis(selected_year, _ver):
    ensure_data_loaded()
    df_d = DATA["daily"]
    df_m = DATA["monthly"]
    try:
        kpi_layout = build_kpi_layout(df_d, df_m, selected_year)
        if df_d.empty:
            prev_tbl = html.Div("(데이터 없음)")
        else:
            years = sorted(df_d['연도'].unique())
            td_year_style = {
                'textAlign': 'left', 'padding': '6px 8px', 'borderBottom': '1px solid #f1f3f5'}
            td_num_style = {'textAlign': 'right', 'padding': '6px 8px', 'borderBottom': '1px solid #f1f3f5',
                            'fontVariantNumeric': 'tabular-nums', 'whiteSpace': 'nowrap'}
            th_year_style = {'textAlign': 'left', 'padding': '6px 8px',
                             'borderBottom': '2px solid #dee2e6', 'color': '#495057'}
            th_num_style = {'textAlign': 'right', 'padding': '6px 8px', 'borderBottom': '2px solid #dee2e6', 'color': '#495057',
                            'fontVariantNumeric': 'tabular-nums', 'whiteSpace': 'nowrap'}
            rows = []
            for y in years:
                if y == selected_year:
                    continue
                tot, avg, days, _, _ = compute_kpis(df_d, y)
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
            prev_tbl = html.Table([header] + rows,
                                  style={
                                      'width': '100%', 'borderCollapse': 'collapse', 'tableLayout': 'fixed'},
                                  className="kpi-table")
        stamp = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")
        return kpi_layout, prev_tbl, f"업데이트: {stamp}"
    except Exception as e:
        return html.Div("KPI 계산 오류"), html.Div(f"오류: {e}"), ""


@callback(
    Output('week-select-fixed', 'options'),
    Output('week-select-fixed', 'value'),
    Input('year-select', 'value'),
    Input('data-version', 'data')
)
def refresh_week_options(_selected_year, _ver):
    ensure_data_loaded()
    return DATA["week_options"], 'this_week'


@callback(
    Output('weekly-chart-today', 'figure'),
    Input('weekly-tabs-today', 'value'),
    Input('data-version', 'data')
)
def refresh_weekly_today(selected_metric, _ver):
    ensure_data_loaded()
    return figure_weekly_today_based(DATA["daily"], value_col=selected_metric)


@callback(
    Output('weekly-chart-fixed', 'figure'),
    Input('week-select-fixed', 'value'),
    Input('weekly-tabs-fixed', 'value'),
    Input('data-version', 'data')
)
def update_week_fixed(monday_str, selected_metric, _ver):
    ensure_data_loaded()
    return figure_weekly_fixed_mon_fri(DATA["daily"], monday_str, value_col=selected_metric)


@callback(
    Output('months-1to12-chart', 'figure'),
    Input('year-select', 'value'),
    Input('data-version', 'data')
)
def refresh_month_chart(_selected_year, _ver):
    ensure_data_loaded()
    cy = datetime.now(KST).year
    return figure_months_1to12(DATA["monthly"], start_year=2022, current_year=cy)


@callback(
    Output("metric-tab-content", "children"),
    Input("metric-tabs", "value"),
    Input("year-select", "value"),
    Input("data-version", "data")
)
def switch_metric_tab(tab_value, selected_year, _ver):
    ensure_data_loaded()
    figs = {
        "avg":   yoy_line_value_bar_rate(DATA["monthly"], '일평균발주량', '월별 일평균 발주량 + YoY%', selected_year),
        "total": yoy_line_value_bar_rate(DATA["monthly"], '발주량',       '월 총 발주량 + YoY%',       selected_year),
        "bw":    yoy_line_value_bar_rate(DATA["monthly"], '흑백출력량',   '월별 흑백 페이지 + YoY%',    selected_year),
        "color": yoy_line_value_bar_rate(DATA["monthly"], '컬러출력량',   '월별 컬러 페이지 + YoY%',    selected_year),
    }
    if tab_value == "forecast":
        layout = forecast_cards_layout(
            DATA["daily"], DATA["monthly"], selected_year)
        return html.Div(style={'padding': '10px'}, children=[layout])
    fig = figs.get(tab_value, go.Figure())
    fig.update_layout(height=460)
    return dcc.Graph(figure=fig, style={'height': '480px'})

# 오늘 패널 갱신


@callback(
    Output('today-floating-panel', 'children'),
    Input('today-refresh', 'n_intervals'),
    Input('data-version', 'data'),
    prevent_initial_call=False
)
def refresh_today_panel(_n, _ver):
    ensure_data_loaded()
    return build_today_panel(DATA["daily"])

# 오늘 패널 표시/숨김


@callback(
    Output('today-floating-panel', 'style'),
    Input('today-visible', 'data'),
    prevent_initial_call=False
)
def apply_today_visibility(visible):
    if bool(visible) or visible is None:
        return {}
    return {'display': 'none'}


@callback(
    Output('today-visible', 'data'),
    Input('fab-today-toggle', 'n_clicks'),
    State('today-visible', 'data'),
    prevent_initial_call=True
)
def toggle_today_visible(_n, visible):
    return not bool(visible)

# 메모 팝업 제어


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
    trig = ctx.triggered[0]['prop_id'].split('.')[0]
    if trig == 'memo-close-btn':
        return False
    return not bool(visible)


@callback(
    Output('memo-popup', 'style'),
    Output('memo-text', 'value'),
    Output('memo-status', 'children'),
    Output('memo-storage', 'data'),
    Input('memo-visible', 'data'),
    Input('memo-save-btn', 'n_clicks'),
    Input('memo-clear-btn', 'n_clicks'),
    State('memo-text', 'value'),
    State('memo-storage', 'data'),
    prevent_initial_call=False
)
def memo_controller(visible, save_n, clear_n, text, stored):
    base_style = {
        'position': 'fixed', 'left': '16px', 'top': '80px',
        'width': 'min(92vw, 360px)', 'zIndex': '1000',
        'background': 'rgba(255,255,255,0.98)', 'backdropFilter': 'blur(2px)',
        'border': '1px solid #edf2f7', 'borderRadius': '14px', 'padding': '10px 12px',
        'boxShadow': '0 10px 22px rgba(0,0,0,0.12)', 'maxHeight': '70vh', 'overflowY': 'auto'
    }
    ctx = dash.callback_context
    if not ctx.triggered:
        display = 'block' if bool(visible) else 'none'
        return ({**base_style, 'display': display}, stored or "", "", stored or "")
    trig = ctx.triggered[0]['prop_id'].split('.')[0]
    if trig == 'memo-visible':
        display = 'block' if bool(visible) else 'none'
        return ({**base_style, 'display': display}, stored if visible else (text or ""), "", stored if visible else (text or ""))
    if trig == 'memo-save-btn':
        content = text or ""
        return ({**base_style, 'display': 'block'}, content, "저장 완료", content)
    if trig == 'memo-clear-btn':
        return ({**base_style, 'display': 'block'}, "", "모든 내용을 지웠습니다.", "")
    display = 'block' if bool(visible) else 'none'
    return ({**base_style, 'display': display}, stored or "", "", stored or "")


# -----------------------------------------------------------------------------
# 로컬 실행
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    host = os.getenv('DASH_HOST', '127.0.0.1')
    port = int(os.getenv('DASH_PORT', '8090'))
    print(f"브라우저에서 http://{host}:{port} 으로 접속하세요.")
    app.run(debug=True, host=host, port=port)

import os
import re
import calendar
from datetime import datetime, timedelta, date
import json

import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials

import dash
from dash import dcc, html, Input, Output, State, callback, no_update
import plotly.express as px
import plotly.graph_objects as go

from dotenv import load_dotenv
import pytz

# -----------------------------------------------------------------------------
# 초기 설정
# -----------------------------------------------------------------------------
load_dotenv()
KST = pytz.timezone('Asia/Seoul')

# -----------------------------------------------------------------------------
# 공통 유틸리티 함수
# -----------------------------------------------------------------------------


def norm(s: str) -> str:
    """문자열에서 공백을 제거하고 정규화합니다."""
    return re.sub(r"\s+", "", (s or "")).strip()


def find_credentials_path():
    """Google 서비스 계정 인증서 경로를 찾거나 환경변수에서 생성합니다."""
    # 환경변수에 지정된 경로 우선
    for p in [os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
              os.getenv("GSPREAD_CREDENTIALS"),
              "credentials.json", "service_account.json"]:
        if p and os.path.exists(p):
            print(f"[INFO] Using credentials file: {p}")
            return p
    # 환경변수에 JSON 내용 전체가 있는 경우
    content = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if content:
        path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "credentials.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(
            f"[INFO] Created credentials file from environment variable -> {path}")
        return path
    raise FileNotFoundError(
        "서비스 계정 JSON 파일이 필요합니다. (credentials.json 또는 GOOGLE_APPLICATION_CREDENTIALS 환경변수)")


def open_sheet():
    """Google Sheets API에 연결하고 지정된 스프레드시트를 엽니다."""
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    creds_file = find_credentials_path()
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_file, scope)
    client = gspread.authorize(creds)

    url = os.getenv("SPREADSHEET_URL", "").strip()
    if not url:
        raise EnvironmentError("SPREADSHEET_URL이 .env 파일이나 환경변수에 설정되지 않았습니다.")
    return client.open_by_url(url)

# -----------------------------------------------------------------------------
# 데이터 로딩 및 정제
# -----------------------------------------------------------------------------


def load_data_from_gsheet():
    """Google 시트에서 일별/월별 데이터를 로드합니다."""
    sh = open_sheet()
    daily_name = os.getenv("DAILY_SHEET_NAME", "일별 발주량 외").strip()
    monthly_name = os.getenv("MONTHLY_SHEET_NAME", "월별 발주량").strip()

    # 일별 데이터 시트
    ws_d = sh.worksheet(daily_name)
    vals_d = ws_d.get_all_values()
    di = int(os.getenv("DAILY_HEADER_INDEX", "0"))
    headers_d = [norm(h) for h in vals_d[di]]
    df_daily_raw = pd.DataFrame(vals_d[di+1:], columns=headers_d)

    # 월별 데이터 시트
    ws_m = sh.worksheet(monthly_name)
    vals_m = ws_m.get_all_values()
    mi = int(os.getenv("MONTHLY_HEADER_INDEX", "1"))
    headers_m = [norm(h) for h in vals_m[mi]]
    df_monthly_raw = pd.DataFrame(vals_m[mi+1:], columns=headers_m)

    return df_daily_raw, df_monthly_raw


def cleanse_daily(df: pd.DataFrame) -> pd.DataFrame:
    """일별 데이터프레임을 정제하고 필요한 컬럼을 생성합니다."""
    if df.empty:
        return pd.DataFrame(columns=['날짜', 'date_only', '연도', '월', '총발주종수', '총발주부수', '흑백페이지', '컬러페이지', '예상제본시간', '최종출고', '출고부수'])

    d = df.copy()
    d = d.rename(columns={c: norm(c) for c in d.columns})

    # 주요 열 이름 추정
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

    # "2025년" 같은 연도 표기 행을 처리하여 날짜 파싱에 활용
    d['__year_header__'] = d[col_date].astype(str).str.extract(
        r'(^\s*(\d{4})\s*년\s*$)', expand=True)[1].ffill()

    # 다양한 날짜 형식을 표준 형식으로 변환
    raw = d[col_date].astype(str).str.strip().str.replace(
        r'\(.+?\)', '', regex=True).str.strip()
    full_pat = re.compile(r'^\d{4}[./-]\d{1,2}[./-]\d{1,2}$')
    partial_pat = re.compile(r'^\d{1,2}[./-]\d{1,2}$')

    def to_full_date(row):
        s = row['__raw__']
        if full_pat.match(s):
            return s.replace('-', '.')
        if partial_pat.match(s) and pd.notna(row['__year_header__']):
            return f"{row['__year_header__']}.{s.replace('-', '.')}"
        return None

    d['__raw__'] = raw
    d['날짜'] = pd.to_datetime(
        d.apply(to_full_date, axis=1), errors='coerce', format='%Y.%m.%d')
    d = d.dropna(subset=['날짜']).copy()

    # 주요 숫자 데이터가 모두 비어있는 행은 제거
    def is_blank(col):
        return d[col].astype(str).str.strip().eq('') if col and col in d.columns else True
    blanks = is_blank(col_cnt) & is_blank(
        col_total) & is_blank(col_bw) & is_blank(col_color)
    d = d.loc[~blanks].copy()

    # 숫자열과 텍스트열 표준화
    def to_num(series):
        return pd.to_numeric(series.astype(str).str.replace(',', '', regex=False), errors='coerce').fillna(0).astype(int)

    def as_text(colname):
        return d[colname].astype(str).str.strip().where(lambda x: x != '', '-') if colname and colname in d.columns else '-'

    d['총발주종수'] = to_num(d[col_cnt])
    d['총발주부수'] = to_num(d[col_total])
    d['흑백페이지'] = to_num(d[col_bw])
    d['컬러페이지'] = to_num(d[col_color])
    d['출고부수'] = to_num(d[col_ships])
    d['예상제본시간'] = as_text(col_bind)
    d['최종출고'] = as_text(col_shipt)

    d['date_only'] = d['날짜'].dt.date
    d['연도'] = d['날짜'].dt.year
    d['월'] = d['날짜'].dt.month

    return d[['날짜', 'date_only', '연도', '월', '총발주종수', '총발주부수', '흑백페이지', '컬러페이지', '예상제본시간', '최종출고', '출고부수']]


def cleanse_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """월별 데이터프레임을 정제합니다."""
    if df.empty:
        return pd.DataFrame(columns=['월DT', '발주량', '발주일수', '일평균발주량', '흑백출력량', '컬러출력량', '연도', '월번호'])

    d = df.copy()
    d = d.rename(columns={c: norm(c) for c in d.columns})
    month_col = '월' if '월' in d.columns else d.columns[0]

    rename_map = {c: norm(c).replace('칼라', '컬러') for c in d.columns}
    d = d.rename(columns=rename_map)

    d['월DT'] = pd.to_datetime(d[month_col].astype(str).str.replace(
        ' ', ''), format='%Y년%m월', errors='coerce')
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
# 데이터 캐시 및 자동 갱신 로직
# -----------------------------------------------------------------------------


def refresh_data_cache():
    """
    Google 시트에서 데이터를 가져와 정제한 후 JSON으로 직렬화하여 반환합니다.
    이 함수는 주기적으로 호출되어 데이터 최신성을 유지합니다.
    """
    print(f"[{datetime.now(KST):%Y-%m-%d %H:%M:%S}] 데이터 갱신 시도...")
    try:
        raw_d, raw_m = load_data_from_gsheet()
        df_d = cleanse_daily(raw_d)
        df_m = cleanse_monthly(raw_m)

        if not df_d.empty:
            years = sorted(df_d['연도'].unique(), reverse=True)
            years_options = [
                {'label': f'{int(y)}년', 'value': int(y)} for y in years]
            week_options = week_options_from_df(df_d)
        else:
            years_options = [
                {'label': f'{datetime.now(KST).year}년', 'value': datetime.now(KST).year}]
            week_options = [{'label': '오늘 기준 (최근 5영업일)', 'value': 'this_week'}]

        data_payload = {
            "daily": df_d.to_json(orient='split', date_format='iso'),
            "monthly": df_m.to_json(orient='split', date_format='iso'),
            "years_options": years_options,
            "week_options": week_options,
            "last_updated": datetime.now(KST).isoformat()
        }
        print(f"[{datetime.now(KST):%Y-%m-%d %H:%M:%S}] 데이터 갱신 성공.")
        return json.dumps(data_payload)

    except Exception as e:
        print(f"[ERROR] 데이터 갱신 실패: {e}")
        return no_update

# -----------------------------------------------------------------------------
# 영업일 및 계절성 관련 계산 함수
# -----------------------------------------------------------------------------
# (이전 코드와 동일, 변경 없음)


def kr_holidays_for_year(year: int):
    try:
        import holidays
        return set(holidays.KR(years=year).keys())
    except ImportError:
        print("[WARN] 'holidays' 라이브러리가 없어 주말만 공휴일로 처리합니다.")
        return set()


def business_days_in_range(start_d: date, end_d: date, holiday_set: set):
    if end_d < start_d:
        return 0
    return sum(1 for i in range((end_d - start_d).days + 1) if (d := start_d + timedelta(days=i)).weekday() < 5 and d not in holiday_set)


def business_day_ratio_for_month(today: date) -> float:
    year, month = today.year, today.month
    start, end_month = date(year, month, 1), date(
        year, month, calendar.monthrange(year, month)[1])
    hol = kr_holidays_for_year(year)
    elapsed = business_days_in_range(start, today, hol)
    total = business_days_in_range(start, end_month, hol)
    return (elapsed / total) if total else 0.0


def seasonal_cum_share_to_date(df_monthly: pd.DataFrame, value_col: str, last_year: int, today: date) -> float:
    d = df_monthly[df_monthly['연도'] == last_year]
    if d.empty or value_col not in d.columns:
        return 0.0
    by_m = d.groupby('월번호')[value_col].sum().reindex(
        range(1, 13), fill_value=0.0)
    total = by_m.sum()
    if total <= 0:
        return 0.0
    shares = by_m / total
    m = today.month
    full_share = shares.loc[1:m-1].sum() if m > 1 else 0.0
    part = shares.loc[m] * business_day_ratio_for_month(today)
    return float(full_share + part)


# -----------------------------------------------------------------------------
# 차트 생성 함수
# -----------------------------------------------------------------------------
# (이전 코드와 거의 동일, 일부 스타일 조정)
WEEKDAY_KR = ['월', '화', '수', '목', '금', '토', '일']


def last_5_business_days_upto_today(now_kst: datetime):
    dates, d, count = [], now_kst.date(), 0
    while count < 5:
        if d.weekday() < 5:
            dates.append(d)
            count += 1
        d -= timedelta(days=1)
    return list(reversed(dates))


def monday(date_obj: date) -> date:
    return date_obj - timedelta(days=date_obj.weekday())


def week_options_from_df(df_daily: pd.DataFrame):
    if df_daily.empty:
        return [{'label': '(데이터 없음)', 'value': 'this_week'}]
    min_d, max_d = df_daily['date_only'].min(), df_daily['date_only'].max()
    opts = []
    cur = monday(max_d)
    while cur >= monday(min_d):
        label = f"{cur:%Y-%m-%d} ~ {(cur + timedelta(days=4)):%Y-%m-%d}"
        opts.append({'label': label, 'value': cur.strftime('%Y-%m-%d')})
        cur -= timedelta(days=7)
    return [{'label': '오늘 기준 (최근 5영업일)', 'value': 'this_week'}] + opts


def figure_weekly_today_based(df_daily: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df_daily.empty:
        fig.update_layout(title='주간 발주량 비교 (데이터 없음)', annotations=[
                          dict(text="데이터가 없습니다", showarrow=False)])
        return fig

    now = datetime.now(KST)
    this_week_dates = last_5_business_days_upto_today(now)
    last_week_dates = [d - timedelta(days=7) for d in this_week_dates]
    m = df_daily.set_index('date_only')['총발주부수'].to_dict()
    y_this = [m.get(d, 0) for d in this_week_dates]
    y_last = [m.get(d, 0) for d in last_week_dates]
    x_week = [WEEKDAY_KR[d.weekday()] for d in this_week_dates]

    fig.add_trace(go.Scatter(x=x_week, y=y_last, mode='lines+markers+text', name='지난 주', line=dict(width=2, dash='dot'), text=[
                  f"{v:,}" if v else "" for v in y_last], textposition='top center', textfont={'size': 11}, hovertemplate="%{customdata|%Y-%m-%d}<br>지난 주: %{y:,}부<extra></extra>", customdata=last_week_dates))
    fig.add_trace(go.Scatter(x=x_week, y=y_this, mode='lines+markers+text', name='이번 주', line=dict(width=3), text=[f"{v:,}" if v else "" for v in y_this], textposition='top center', textfont={
                  'size': 11}, hovertemplate="%{customdata|%Y-%m-%d}<br>이번 주: %{y:,}부<extra></extra>", customdata=this_week_dates))
    fig.update_layout(title=f'주간 발주량 비교 (기준일: {now:%Y-%m-%d})', xaxis_title='', yaxis_title='발주 부수', template='plotly_white',
                      height=280, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation='h', x=1, xanchor='right', y=1.1))
    return fig

# (figure_weekly_fixed_mon_fri, figure_months_1to12, yoy_line_value_bar_rate 등 다른 차트 함수는 이전과 동일)


def figure_weekly_fixed_mon_fri(df_daily: pd.DataFrame, monday_str: str = None) -> go.Figure:
    fig = go.Figure()
    if df_daily.empty:
        return fig.update_layout(title='주간 발주량 비교 (데이터 없음)')

    base_mon = datetime.strptime(monday_str, "%Y-%m-%d").date(
    ) if monday_str and monday_str != 'this_week' else monday(datetime.now(KST).date())
    week_days = [base_mon + timedelta(days=i) for i in range(5)]
    prev_week_days = [d - timedelta(days=7) for d in week_days]

    m = df_daily.set_index('date_only')['총발주부수'].to_dict()
    y_this = [m.get(d, 0) for d in week_days]
    y_last = [m.get(d, 0) for d in prev_week_days]
    x_week = [WEEKDAY_KR[d.weekday()] for d in week_days]

    fig.add_trace(go.Scatter(x=x_week, y=y_last, mode='lines+markers+text', name='지난 주', line=dict(width=2, dash='dot'), text=[
                  f"{v:,}" if v else "" for v in y_last], textposition='top center', textfont={'size': 11}, hovertemplate="%{customdata|%Y-%m-%d}<br>지난 주: %{y:,}부<extra></extra>", customdata=prev_week_days))
    fig.add_trace(go.Scatter(x=x_week, y=y_this, mode='lines+markers+text', name='이번 주', line=dict(width=3), text=[f"{v:,}" if v else "" for v in y_this], textposition='top center', textfont={
                  'size': 11}, hovertemplate="%{customdata|%Y-%m-%d}<br>이번 주: %{y:,}부<extra></extra>", customdata=week_days))
    fig.update_layout(title=f"주간 발주량 비교 (월~금): {base_mon:%Y-%m-%d} ~ {week_days[-1]:%Y-%m-%d}", xaxis_title='', yaxis_title='발주 부수',
                      template='plotly_white', height=300, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation='h', x=1, xanchor='right', y=1.1))
    return fig


def figure_months_1to12(df_monthly: pd.DataFrame, start_year=2022, current_year=None) -> go.Figure:
    fig = go.Figure()
    if df_monthly.empty:
        return fig.update_layout(title='월별 발주량 (데이터 없음)')
    if current_year is None:
        current_year = datetime.now(KST).year
    d = df_monthly[(df_monthly['연도'] >= start_year) &
                   (df_monthly['연도'] <= current_year)]
    if d.empty:
        return fig
    pivot = d.pivot_table(index='월번호', columns='연도',
                          values='발주량', aggfunc='sum').sort_index()
    fig = px.line(pivot, x=pivot.index, y=pivot.columns, markers=True,
                  title=f'월별 발주량 (1~12월, {start_year}~{current_year})')
    fig.update_layout(xaxis_title='', yaxis_title='발주량', template='plotly_white', height=300, margin=dict(
        l=20, r=20, t=40, b=20), legend=dict(orientation='h', x=1, xanchor='right', y=1.1), xaxis=dict(dtick=1))
    return fig


def yoy_line_value_bar_rate(d: pd.DataFrame, value_col: str, title: str, baseline_year: int) -> go.Figure:
    fig = go.Figure()
    if d.empty:
        return fig.update_layout(title=f'{title} (데이터 없음)')
    d = d[d['연도'] <= baseline_year].sort_values(['연도', '월번호'])
    if value_col not in d.columns:
        d[value_col] = 0
    d['prev_year'] = d.groupby('월번호')[value_col].shift(1)
    d['YoY%'] = ((d[value_col] - d['prev_year']) /
                 d['prev_year'].replace({0: pd.NA})) * 100
    for y, sub in d.groupby('연도'):
        fig.add_trace(go.Scatter(
            x=sub['월번호'], y=sub[value_col], mode='lines+markers', name=f'{y}년 ({value_col})'))
    base = d[d['연도'] == baseline_year]
    fig.add_trace(go.Bar(x=base['월번호'], y=base['YoY%'], name=f'{baseline_year} YoY%',
                  yaxis='y2', opacity=0.6, hovertemplate="증감율: %{y:.1f}%<extra></extra>"))
    fig.update_layout(title=title, xaxis_title='', yaxis_title=value_col, yaxis2=dict(title='YoY %', overlaying='y', side='right', showgrid=False),
                      template='plotly_white', height=420, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation='h', x=1, xanchor='right', y=1.12), xaxis=dict(dtick=1))
    return fig

# -----------------------------------------------------------------------------
# KPI 및 컴포넌트 생성 함수
# -----------------------------------------------------------------------------


def compute_kpis(df_daily: pd.DataFrame, year: int):
    if df_daily.empty:
        return 0, 0.0, 0, 0, 0
    year_df = df_daily[df_daily['연도'] == year]
    if year_df.empty:
        return 0, 0.0, 0, 0, 0
    total_orders = int(year_df['총발주부수'].sum())
    days_count = int(year_df['date_only'].nunique())
    avg_per_day = round(total_orders / days_count, 2) if days_count else 0.0
    total_bw_pages = int(year_df['흑백페이지'].sum())
    total_color_pages = int(year_df['컬러페이지'].sum())
    return total_orders, avg_per_day, days_count, total_bw_pages, total_color_pages


def compute_progress_advanced(df_daily: pd.DataFrame, df_monthly: pd.DataFrame, year: int):
    today = datetime.now(KST).date()
    ly = year - 1
    ytd_curr = df_daily[(df_daily['연도'] == year) & (
        df_daily['date_only'] <= today)]['총발주부수'].sum()
    try:
        same_date_ly = today.replace(year=ly)
    except ValueError:
        same_date_ly = date(ly, 2, 28)
    ytd_ly = df_daily[(df_daily['연도'] == ly) & (
        df_daily['date_only'] <= same_date_ly)]['총발주부수'].sum()
    last_year_total = df_daily[df_daily['연도'] == ly]['총발주부수'].sum()
    hol = kr_holidays_for_year(year)
    elapsed_biz_days = business_days_in_range(date(year, 1, 1), today, hol)
    total_biz_days = business_days_in_range(
        date(year, 1, 1), date(year, 12, 31), hol)
    ratio_biz = (elapsed_biz_days / total_biz_days) if total_biz_days else 0.0
    target_biz = last_year_total * ratio_biz
    progress_vs_biz = (ytd_curr / target_biz * 100) if target_biz else None
    seasonal_share = seasonal_cum_share_to_date(df_monthly, '발주량', ly, today)
    target_seasonal = last_year_total * seasonal_share
    progress_vs_seasonal = (ytd_curr / target_seasonal *
                            100) if target_seasonal else None
    return {'ytd_curr': int(ytd_curr), 'ytd_ly': int(ytd_ly), 'last_year_total': int(last_year_total), 'ratio_biz': ratio_biz, 'progress_vs_biz': progress_vs_biz, 'seasonal_share_to_date': seasonal_share, 'progress_vs_seasonal': progress_vs_seasonal, 'elapsed_biz_days': elapsed_biz_days, 'total_biz_days': total_biz_days, 'today': today, 'same_date_ly': same_date_ly}


def kpi_card(title, value, subtitle):
    return html.Div([
        html.Div(title, style={'fontSize': '0.95rem', 'color': '#666',
                 'marginBottom': '6px', 'fontWeight': '600'}),
        html.Div(value, style={'fontSize': '1.8rem', 'fontWeight': '800'}),
        html.Div(subtitle, style={'fontSize': '0.85rem',
                 'color': '#8a8a8a', 'marginTop': '4px'}),
    ], style={'flex': '1 1 260px', 'background': 'white', 'borderRadius': '14px', 'padding': '14px 16px', 'boxShadow': '0 6px 18px rgba(0,0,0,0.08)', 'minWidth': '240px'})


def build_kpi_layout(df_daily: pd.DataFrame, df_monthly: pd.DataFrame, year: int):
    # (이전 코드와 동일, 변경 없음)
    tot, avg, days, bw, color = compute_kpis(df_daily, year)
    row1 = html.Div([kpi_card(f"{year}년 총 발주량", f"{tot:,}", "Total Orders"), kpi_card(f"{year}년 일 평균 발주량", f"{avg:,}", "Avg / Working Day"), kpi_card(
        f"{year}년 총 발주 일수", f"{days:,}일", "Working Days Count")], style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap', 'marginBottom': '8px'})
    row2 = html.Div([kpi_card(f"{year}년 흑백 페이지 합계", f"{bw:,}", "BW Pages (Daily sum)"), kpi_card(
        f"{year}년 컬러 페이지 합계", f"{color:,}", "Color Pages (Daily sum)")], style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap', 'marginBottom': '6px'})
    prog = compute_progress_advanced(df_daily, df_monthly, year)
    badges = []
    if prog['ytd_ly']:
        badges.append(html.Span(f"YTD vs 작년 동기간: {prog['ytd_curr'] / prog['ytd_ly'] * 100:.1f}%",
                      title=f"올해 YTD({prog['today']:%m/%d}) {prog['ytd_curr']:,} ÷ 작년 YTD({prog['same_date_ly']:%m/%d}) {prog['ytd_ly']:,}", className="kpi-badge blue"))
    if prog['progress_vs_biz'] is not None:
        badges.append(html.Span(f"경과율(영업일) 대비: {prog['progress_vs_biz']:.1f}%", title=f"올해 YTD {prog['ytd_curr']:,} ÷ (작년 연간 {prog['last_year_total']:,} × 영업일 경과율 {prog['ratio_biz']*100:.1f}%)",
                      className=f"kpi-badge {'green' if prog['progress_vs_biz'] >= 100 else 'red'}"))
    if prog['progress_vs_seasonal'] is not None:
        badges.append(html.Span(f"월별 가중치 대비: {prog['progress_vs_seasonal']:.1f}%", title=f"올해 YTD {prog['ytd_curr']:,} ÷ (작년 연간 {prog['last_year_total']:,} × 월별 가중 누적비중 {prog['seasonal_share_to_date']*100:.1f}%)",
                      className=f"kpi-badge {'green' if prog['progress_vs_seasonal'] >= 100 else 'red'}"))
    badges.append(html.Span(f"영업일 경과율: {prog['ratio_biz']*100:.1f}%",
                  title=f"올해 1/1~{prog['today']:%m/%d} 영업일 {prog['elapsed_biz_days']}/{prog['total_biz_days']}일", className="kpi-badge gray"))
    badges.append(html.Span(
        f"월별 가중 누적비중: {prog['seasonal_share_to_date']*100:.1f}%", title="작년 월별 연간 비중 누적치", className="kpi-badge gray"))
    row3 = html.Div(badges, style={'display': 'flex', 'gap': '8px',
                    'flexWrap': 'wrap', 'alignItems': 'center', 'margin': '4px 2px 0'})
    return html.Div([row1, row2, row3])


def build_today_panel_content(df_daily: pd.DataFrame):
    """'오늘 현황' 팝업의 내용을 생성합니다."""
    today = datetime.now(KST).date()
    if df_daily.empty:
        return html.Div("데이터를 불러오는 중입니다...", style={'color': '#666'})

    row = df_daily[df_daily['date_only'] == today]
    if row.empty:
        return html.Div("오늘 데이터가 아직 없습니다.", style={'color': '#666', 'padding': '20px 0'})

    r = row.iloc[0]
    def fmt(x): return f"{int(x):,}" if pd.notna(
        x) and isinstance(x, (int, float)) else (str(x) or '-')

    items = {
        "총 발주 종수": fmt(r.get('총발주종수')), "총 발주 부수": fmt(r.get('총발주부수')),
        "흑백 페이지": fmt(r.get('흑백페이지')), "컬러 페이지": fmt(r.get('컬러페이지')),
        "예상 제본 시간": fmt(r.get('예상제본시간')), "최종 출고": fmt(r.get('최종출고')),
        "출고 부수": fmt(r.get('출고부수'))
    }
    grid = [html.Div(k, style={'color': '#666'}), html.Div(v, style={'textAlign': 'right', 'fontWeight': '700'}) for k, v in items.items()]
    return html.Div(grid, style={'display': 'grid', 'gridTemplateColumns': 'auto 1fr', 'rowGap': '8px', 'columnGap': '16px'})


def forecast_cards_layout(df_daily, df_monthly, year):
    """연간 예측 탭의 플레이스홀더 레이아웃입니다."""
    return html.Div("연간 예측 기능은 현재 준비 중입니다.", style={'padding': '30px', 'textAlign': 'center', 'color': '#888'})


# -----------------------------------------------------------------------------
# Dash 앱 레이아웃
# -----------------------------------------------------------------------------
external_stylesheets = [
    {"href": "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css", "rel": "stylesheet"}]
app = dash.Dash(__name__, title="발주량 분석 대시보드", meta_tags=[
                {"name": "viewport", "content": "width=device-width, initial-scale=1"}], external_stylesheets=external_stylesheets)
server = app.server
CURRENT_YEAR = datetime.now(KST).year

app.layout = html.Div([
    # 데이터 저장소 및 주기적 실행 컴포넌트
    dcc.Store(id='data-store'),  # 모든 데이터를 JSON으로 저장
    dcc.Store(id='memo-storage', storage_type='local'),  # 메모장 내용 영구 저장
    dcc.Interval(id='data-refresh-interval', interval=10 *
                 60 * 1000, n_intervals=0),  # 10분마다 데이터 갱신
    dcc.Interval(id='today-panel-interval', interval=2 *
                 60 * 1000, n_intervals=0),  # 2분마다 오늘 현황 갱신

    # 메인 콘텐츠 영역
    html.Div(style={'maxWidth': '1100px', 'margin': '0 auto', 'padding': '16px 20px 80px 20px'}, children=[
        html.H1("발주량 분석 대시보드", style={
                'textAlign': 'center', 'marginBottom': '6px'}),
        html.P([
            "구글 시트 데이터 기반 · 2021~현재 · ",
            html.Span(id='last-updated-text', style={'color': '#888'})
        ], style={'textAlign': 'center', 'marginBottom': '14px', 'color': '#666'}),

        # KPI 섹션
        html.Div([
            html.Span("KPI 연도 선택:", style={'fontWeight': '600'}),
            dcc.Dropdown(id='year-select', options=[{'label': f'{CURRENT_YEAR}년', 'value': CURRENT_YEAR}],
                         value=CURRENT_YEAR, clearable=False, style={'width': '220px'}),
        ], style={'display': 'flex', 'gap': '10px', 'justifyContent': 'center', 'alignItems': 'center', 'marginBottom': '8px'}),
        html.Div(id='kpi-cards', children=[html.Div("데이터 로딩 중...",
                 style={'textAlign': 'center', 'padding': '50px'})]),
        html.Details([
            html.Summary("지난 연도 KPI 펼치기 / 접기"),
            html.Div(id='prev-years-kpi', style={'marginTop': '8px'})
        ], style={'background': 'white', 'borderRadius': '12px', 'padding': '14px', 'boxShadow': '0 4px 14px rgba(0,0,0,0.08)', 'marginBottom': '16px'}),

        # 차트 섹션
        html.Div(className='chart-container', children=[
            html.H3("주간 발주량 비교 (오늘 기준 5영업일 + 지난주 같은 요일)",
                    className='chart-title'),
            dcc.Graph(id='weekly-chart-today', style={'height': '300px'})
        ]),
        html.Details([
            html.Summary("월~금 고정 주간 비교 (클릭하여 열기)"),
            html.Div([
                html.Span("주차 선택:", style={'fontWeight': '600'}),
                dcc.Dropdown(id='week-select-fixed', value='this_week',
                             clearable=False, style={'width': '300px'}),
            ], style={'display': 'flex', 'gap': '8px', 'alignItems': 'center', 'margin': '8px 0'}),
            dcc.Graph(id='weekly-chart-fixed', style={'height': '320px'})
        ], className='chart-container-details'),
        html.Div(className='chart-container', children=[
            html.H3("월별 발주량 (1~12월, 2022~현재)", className='chart-title'),
            dcc.Graph(id='months-1to12-chart', style={'height': '320px'})
        ]),
        html.Div(className='chart-container', style={'padding': '0'}, children=[
            dcc.Tabs(id="metric-tabs", value="avg", children=[
                dcc.Tab(label="일평균 발주량", value="avg"), dcc.Tab(
                    label="월 총 발주량", value="total"),
                dcc.Tab(label="흑백 페이지", value="bw"), dcc.Tab(
                    label="컬러 페이지", value="color"),
                dcc.Tab(label="연간 예측", value="forecast"),
            ]),
            html.Div(id="metric-tab-content", style={'padding': '8px 4px'})
        ]),
    ]),

    # 하단 고정 버튼 (FAB)
    html.Button(html.I(className="fa-solid fa-note-sticky"),
                id="open-memo-modal", className="fab fab-left", title="메모장 열기"),
    html.Button(html.I(className="fa-solid fa-calendar-day"),
                id="open-today-modal", className="fab fab-right", title="오늘 현황 열기"),

    # 모달 (팝업) 영역
    html.Div(id='today-modal', className='modal-container', style={'display': 'none'}, children=[
        html.Div(className='modal-content', children=[
            html.Div([
                html.H3("오늘 현황"),
                html.Button(html.I(className="fa-solid fa-xmark"),
                            id='close-today-modal', className='modal-close-btn')
            ], className='modal-header'),
            html.Div(id='today-modal-content', className='modal-body')
        ])
    ]),
    html.Div(id='memo-modal', className='modal-container', style={'display': 'none'}, children=[
        html.Div(className='modal-content', children=[
            html.Div([
                html.H3("메모장"),
                html.Button(html.I(className="fa-solid fa-xmark"),
                            id='close-memo-modal', className='modal-close-btn')
            ], className='modal-header'),
            dcc.Textarea(
                id='memo-textarea', placeholder="여기에 메모를 입력하세요...", className='memo-textarea'),
            html.Div([
                html.Button("지우기", id="clear-memo",
                            className="memo-btn memo-btn-clear"),
                html.Button("저장", id="save-memo",
                            className="memo-btn memo-btn-save")
            ], className='memo-btn-container')
        ])
    ]),
])

# -----------------------------------------------------------------------------
# 콜백 (앱의 동적 기능)
# -----------------------------------------------------------------------------

# 1. 데이터 자동 갱신 및 초기 로드


@callback(
    Output('data-store', 'data'),
    Input('data-refresh-interval', 'n_intervals')
)
def update_data_store(_):
    return refresh_data_cache()

# 2. 데이터 로드 후 UI 업데이트 (연도/주차 선택 옵션 등)


@callback(
    Output('year-select', 'options'),
    Output('year-select', 'value'),
    Output('week-select-fixed', 'options'),
    Output('last-updated-text', 'children'),
    Input('data-store', 'data')
)
def update_dropdowns_from_data(stored_data):
    if not stored_data:
        return no_update, no_update, no_update, "데이터 로딩 중..."

    data = json.loads(stored_data)
    years_opts = data.get('years_options', [])
    latest_year = years_opts[0]['value'] if years_opts else CURRENT_YEAR
    week_opts = data.get('week_options', [])
    last_updated = f"마지막 업데이트: {datetime.fromisoformat(data['last_updated']).strftime('%H:%M:%S')}"

    return years_opts, latest_year, week_opts, last_updated

# 3. KPI 카드 및 관련 차트 업데이트


@callback(
    Output('kpi-cards', 'children'),
    Output('prev-years-kpi', 'children'),
    Output('weekly-chart-today', 'figure'),
    Output('months-1to12-chart', 'figure'),
    Input('year-select', 'value'),
    State('data-store', 'data')
)
def update_main_kpis_and_charts(selected_year, stored_data):
    if not stored_data or not selected_year:
        return [html.Div("연도를 선택하세요.")] * 4

    data = json.loads(stored_data)
    df_d = pd.read_json(data['daily'], orient='split')
    df_d['date_only'] = pd.to_datetime(df_d['date_only'], unit='ms').dt.date
    df_m = pd.read_json(data['monthly'], orient='split')

    kpi_layout = build_kpi_layout(df_d, df_m, selected_year)

    # 지난 연도 KPI 테이블
    years = sorted(df_d['연도'].unique(), reverse=True)
    rows = []
    for y in years:
        if y == selected_year:
            continue
        tot, avg, days, _, _ = compute_kpis(df_d, y)
        rows.append(html.Tr([html.Td(f"{y}년"), html.Td(
            f"{tot:,}"), html.Td(f"{avg:,}"), html.Td(f"{days:,}")]))
    prev_tbl = html.Table([html.Thead(html.Tr([html.Th("연도"), html.Th("총 발주량"), html.Th(
        "일 평균 발주량"), html.Th("총 발주 일수")]))] + [html.Tbody(rows)], className="kpi-table")

    # 차트 생성
    fig_weekly = figure_weekly_today_based(df_d)
    fig_monthly = figure_months_1to12(
        df_m, start_year=2022, current_year=datetime.now(KST).year)

    return kpi_layout, prev_tbl, fig_weekly, fig_monthly

# 4. 탭 기반 차트 업데이트


@callback(
    Output("metric-tab-content", "children"),
    Input("metric-tabs", "value"),
    State("year-select", "value"),
    State('data-store', 'data')
)
def switch_metric_tab(tab_value, selected_year, stored_data):
    if not stored_data:
        return "데이터 로딩 중..."

    data = json.loads(stored_data)
    df_m = pd.read_json(data['monthly'], orient='split')

    if tab_value == "forecast":
        df_d = pd.read_json(data['daily'], orient='split')
        df_d['date_only'] = pd.to_datetime(
            df_d['date_only'], unit='ms').dt.date
        return forecast_cards_layout(df_d, df_m, selected_year)

    col_map = {"avg": "일평균발주량", "total": "발주량",
               "bw": "흑백출력량", "color": "컬러출력량"}
    title_map = {"avg": "월별 일평균 발주량", "total": "월 총 발주량",
                 "bw": "월별 흑백 페이지", "color": "월별 컬러 페이지"}
    fig = yoy_line_value_bar_rate(
        df_m, col_map[tab_value], f'{title_map[tab_value]} + YoY%', selected_year)
    fig.update_layout(height=460)
    return dcc.Graph(figure=fig, style={'height': '480px'})

# 5. 고정 주간 차트 업데이트


@callback(
    Output('weekly-chart-fixed', 'figure'),
    Input('week-select-fixed', 'value'),
    State('data-store', 'data')
)
def update_week_fixed(monday_str, stored_data):
    if not stored_data:
        return go.Figure()
    data = json.loads(stored_data)
    df_d = pd.read_json(data['daily'], orient='split')
    df_d['date_only'] = pd.to_datetime(df_d['date_only'], unit='ms').dt.date
    return figure_weekly_fixed_mon_fri(df_d, monday_str)

# 6. 모달 열기/닫기 콜백


@callback(
    Output('today-modal', 'style'),
    Output('memo-modal', 'style'),
    Input('open-today-modal', 'n_clicks'),
    Input('close-today-modal', 'n_clicks'),
    Input('open-memo-modal', 'n_clicks'),
    Input('close-memo-modal', 'n_clicks'),
    State('today-modal', 'style'),
    State('memo-modal', 'style'),
    prevent_initial_call=True
)
def toggle_modals(n_open_today, n_close_today, n_open_memo, n_close_memo, style_today, style_memo):
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update, no_update

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id in ['open-today-modal', 'close-today-modal']:
        new_style = {'display': 'none'} if style_today.get(
            'display') != 'none' else {'display': 'flex'}
        return new_style, no_update

    if button_id in ['open-memo-modal', 'close-memo-modal']:
        new_style = {'display': 'none'} if style_memo.get(
            'display') != 'none' else {'display': 'flex'}
        return no_update, new_style

    return no_update, no_update

# 7. '오늘 현황' 패널 내용 주기적 갱신


@callback(
    Output('today-modal-content', 'children'),
    Input('today-panel-interval', 'n_intervals'),
    State('data-store', 'data')
)
def refresh_today_panel_content(_, stored_data):
    if not stored_data:
        return "데이터 로딩 중..."
    data = json.loads(stored_data)
    df_d = pd.read_json(data['daily'], orient='split')
    df_d['date_only'] = pd.to_datetime(df_d['date_only'], unit='ms').dt.date
    return build_today_panel_content(df_d)

# 8. 메모장 기능 콜백


@callback(
    Output('memo-storage', 'data'),
    Input('save-memo', 'n_clicks'),
    Input('clear-memo', 'n_clicks'),
    State('memo-textarea', 'value'),
    prevent_initial_call=True,
)
def handle_memo_buttons(save_clicks, clear_clicks, memo_text):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'save-memo':
        return memo_text
    if button_id == 'clear-memo':
        return ''
    return no_update


@callback(
    Output('memo-textarea', 'value'),
    Input('memo-storage', 'data')
)
def load_memo_from_storage(memo_data):
    return memo_data or ''


# -----------------------------------------------------------------------------
# 로컬 개발 환경에서 앱 실행
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # 외부 CSS 파일 추가
    with open("assets/styles.css", "w") as f:
        f.write("""
body { font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif; background-color: #f8f9fa; }
.chart-container { background: white; border-radius: 12px; padding: 14px; box-shadow: 0 4px 14px rgba(0,0,0,0.08); margin-bottom: 16px; }
.chart-container-details { background: white; border-radius: 12px; padding: 14px; box-shadow: 0 4px 14px rgba(0,0,0,0.08); margin-bottom: 12px; }
.chart-title { margin-bottom: 4px; font-size: 1.05rem; }
.kpi-table { width: 100%; border-collapse: collapse; table-layout: fixed; font-size: 0.9rem; }
.kpi-table th, .kpi-table td { padding: 8px; border-bottom: 1px solid #f1f3f5; text-align: right; }
.kpi-table th { font-weight: 600; border-bottom: 2px solid #dee2e6; color: #495057; }
.kpi-table td:first-child, .kpi-table th:first-child { text-align: left; }
.kpi-badge { display: inline-block; padding: 4px 10px; border-radius: 999px; color: white; font-size: 0.8rem; font-weight: 700; cursor: help; }
.kpi-badge.blue { background-color: #1c7ed6; }
.kpi-badge.green { background-color: #2f9e44; }
.kpi-badge.red { background-color: #e03131; }
.kpi-badge.gray { background-color: #868e96; }
.fab { position: fixed; bottom: 20px; width: 56px; height: 56px; border-radius: 50%; border: none; color: white; font-size: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.2); cursor: pointer; display: flex; align-items: center; justify-content: center; z-index: 1000; transition: transform 0.2s ease; }
.fab:hover { transform: scale(1.1); }
.fab-left { left: 20px; background-color: #fab005; }
.fab-right { right: 20px; background-color: #15aabf; }
.modal-container { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.5); z-index: 1001; display: flex; align-items: center; justify-content: center; }
.modal-content { background: white; border-radius: 16px; padding: 20px; box-shadow: 0 8px 24px rgba(0,0,0,0.15); width: 90%; max-width: 500px; display: flex; flex-direction: column; max-height: 80vh; }
.modal-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #e9ecef; padding-bottom: 12px; margin-bottom: 16px; }
.modal-header h3 { margin: 0; }
.modal-close-btn { background: none; border: none; font-size: 20px; cursor: pointer; color: #868e96; padding: 4px; }
.modal-body { overflow-y: auto; }
.memo-textarea { width: 100%; height: 200px; border: 1px solid #ced4da; border-radius: 8px; padding: 10px; font-size: 1rem; resize: vertical; margin-bottom: 12px; }
.memo-btn-container { display: flex; justify-content: flex-end; gap: 8px; }
.memo-btn { padding: 8px 16px; border-radius: 8px; border: none; font-weight: 600; cursor: pointer; }
.memo-btn-save { background-color: #228be6; color: white; }
.memo-btn-clear { background-color: #e9ecef; color: #495057; }
        """)
    if not os.path.exists("assets"):
        os.makedirs("assets")
    os.rename("assets/styles.css", "assets/styles.css")

    host = os.getenv('DASH_HOST', '127.0.0.1')
    port = int(os.getenv('DASH_PORT', '8090'))
    print(f"Dash 앱을 시작합니다. 브라우저에서 http://{host}:{port} 으로 접속하세요.")
    app.run(debug=True, host=host, port=port)

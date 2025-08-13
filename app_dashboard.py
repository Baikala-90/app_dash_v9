import os
import json
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from dotenv import load_dotenv

# --- 1. 환경 설정 및 초기화 ---
load_dotenv()
app = dash.Dash(__name__, external_stylesheets=[
                dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server

# --- 2. Google Sheets 데이터 로딩 및 처리 ---


def load_and_process_data():
    """Google Sheets에서 데이터를 로드하고 Pandas DataFrame으로 정제합니다."""
    try:
        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name(
            'credentials.json', scope)
        client = gspread.authorize(creds)
        sheet_url = os.getenv(
            'SPREADSHEET_URL', "https://docs.google.com/spreadsheets/d/1te1Z9YDH7ThzAwXTJuULeXo_fvgUbiiEs_8qVKEFza8/edit?gid=0#gid=0")
        sheet = client.open_by_url(sheet_url)

        # 일별 데이터 로드 (헤더: 3행)
        daily_worksheet = sheet.worksheet("일별 발주량 외")
        daily_data = daily_worksheet.get_all_values()
        df_daily = pd.DataFrame(daily_data[3:], columns=daily_data[2])
        df_daily = df_daily[df_daily['날짜'] != ''].copy()
        df_daily['날짜'] = pd.to_datetime(df_daily['날짜'], errors='coerce')
        df_daily.dropna(subset=['날짜'], inplace=True)
        for col in ['총 발주 부수', '흑백페이지', '컬러페이지']:
            df_daily[col] = pd.to_numeric(df_daily[col].astype(
                str).str.replace(',', ''), errors='coerce').fillna(0)

        # 월별 데이터 로드 (헤더: 2행)
        monthly_worksheet = sheet.worksheet("월별 발주량")
        monthly_data = monthly_worksheet.get_all_values()
        df_monthly = pd.DataFrame(monthly_data[2:], columns=monthly_data[1])
        df_monthly = df_monthly[df_monthly['월'] != ''].copy()
        df_monthly['월'] = pd.to_datetime(df_monthly['월'].str.replace(
            ' ', ''), format='%Y년%m월', errors='coerce')
        df_monthly.dropna(subset=['월'], inplace=True)
        for col in ['발주량', '발주일수', '일 평균 발주량', '흑백출력량', '컬러 출력량']:
            df_monthly[col] = pd.to_numeric(df_monthly[col].astype(
                str).str.replace(',', ''), errors='coerce').fillna(0)

        # 2022년 이후 데이터만 필터링
        df_monthly = df_monthly[df_monthly['월'].dt.year >= 2022].copy()

        return df_daily, df_monthly
    except Exception as e:
        print(f"데이터 로딩 오류: {e}")
        return pd.DataFrame(), pd.DataFrame()

# --- 3. 데이터 및 차트 생성 함수 ---


def create_empty_fig(message="데이터를 불러올 수 없습니다."):
    """데이터 로드 실패 시 빈 그래프를 생성합니다."""
    fig = go.Figure()
    fig.update_layout(
        xaxis={'visible': False},
        yaxis={'visible': False},
        annotations=[{'text': message, 'xref': 'paper',
                      'yref': 'paper', 'showarrow': False, 'font': {'size': 16}}]
    )
    return fig


def create_weekly_chart(df_daily):
    """요청사항 1: 최근 5영업일 기준 주간 비교 꺾은선 그래프 생성"""
    if df_daily.empty or len(df_daily) < 10:
        return create_empty_fig("주간 비교를 위한 데이터가 부족합니다.")

    df_workdays = df_daily[df_daily['날짜'].dt.weekday <
                           5].sort_values('날짜', ascending=False)
    this_week_df = df_workdays.head(5).sort_values('날짜')
    last_week_df = df_workdays.iloc[5:10].sort_values('날짜')

    this_week_df['요일'] = this_week_df['날짜'].dt.strftime('%a')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=this_week_df['요일'], y=this_week_df['총 발주 부수'],
        mode='lines+markers', name='최근 5일', text=this_week_df['날짜'].dt.strftime('%m-%d'),
        hoverinfo='name+y+text', line=dict(color='#0d6efd', width=3), marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=this_week_df['요일'], y=last_week_df['총 발주 부수'].values,
        mode='lines+markers', name='이전 5일', text=last_week_df['날짜'].dt.strftime('%m-%d'),
        hoverinfo='name+y+text', line=dict(color='#ff7f0e', width=3, dash='dash'), marker=dict(size=8)
    ))
    fig.update_layout(title='주간 발주량 비교 (최근 5영업일)',
                      template='plotly_white', margin=dict(t=50, b=50, l=50, r=30))
    return fig


def create_multi_year_monthly_trend_chart(df_monthly):
    """요청사항 2: 2022년부터 현재까지 연도별 월간 발주량 꺾은선 그래프"""
    if df_monthly.empty:
        return create_empty_fig()

    df_monthly['연도'] = df_monthly['월'].dt.year
    df_monthly['월별'] = df_monthly['월'].dt.month

    fig = go.Figure()
    for year in sorted(df_monthly['연도'].unique()):
        year_data = df_monthly[df_monthly['연도'] == year].sort_values('월별')
        fig.add_trace(go.Scatter(
            x=year_data['월별'], y=year_data['발주량'], mode='lines+markers', name=f'{year}년'))

    fig.update_layout(title='연도별 월간 발주량 추이 (2022-현재)', template='plotly_white', margin=dict(t=50, b=50, l=50, r=30),
                      xaxis=dict(tickmode='array', tickvals=list(range(1, 13)), title='월'))
    return fig


def create_detailed_monthly_chart(df_monthly, value_col, title):
    """요청사항 3: 월별 상세 지표 비교 막대그래프 (증감률 포함)"""
    if df_monthly.empty:
        return create_empty_fig()

    df_monthly['연도'] = df_monthly['월'].dt.year
    df_monthly['월별'] = df_monthly['월'].dt.month
    current_year = datetime.now().year

    df_pivot = df_monthly.pivot_table(
        index='월별', columns='연도', values=value_col, aggfunc='sum')

    fig = go.Figure()
    for year in sorted(df_pivot.columns):
        fig.add_trace(
            go.Bar(x=df_pivot.index, y=df_pivot[year], name=f'{year}년'))

    if (current_year - 1) in df_pivot.columns and current_year in df_pivot.columns:
        last_year_vals = df_pivot[current_year - 1]
        current_year_vals = df_pivot[current_year]
        growth = ((current_year_vals - last_year_vals) /
                  last_year_vals.replace(0, pd.NA) * 100).fillna(0)
        growth_text = [f"{g:+.1f}%" if g != 0 else "" for g in growth]
        fig.add_trace(go.Scatter(
            x=df_pivot.index, y=df_pivot[current_year], mode='text', text=growth_text,
            textposition='top center', textfont=dict(size=10, color='crimson'), showlegend=False
        ))

    fig.update_layout(title=f'{title} (전년 동월 대비 증감률)', template='plotly_white', barmode='group',
                      margin=dict(t=50, b=50, l=50, r=30), xaxis=dict(tickmode='array', tickvals=list(range(1, 13)), title='월'))
    return fig


# --- 4. 데이터 로드 및 대시보드 레이아웃 ---
df_daily, df_monthly = load_and_process_data()

app.layout = dbc.Container([
    # 헤더
    dbc.Row(dbc.Col(html.H1("발주량 분석 대시보드", className="text-center my-4"), width=12)),

    # KPI 섹션
    dbc.Card([
        dbc.CardHeader([
            dbc.Row([
                dbc.Col(html.H4("핵심 성과 지표 (KPI)", className="mb-0"), width="auto"),
                dbc.Col(
                    dbc.ButtonGroup([
                        dbc.Button("올해", id="btn-this-year",
                                   color="primary", n_clicks=1),
                        dbc.Button("작년", id="btn-last-year",
                                   color="secondary", n_clicks=0),
                    ], className="ms-auto")
                )
            ], align="center")
        ]),
        dbc.CardBody(id="kpi-cards")
    ], className="mb-4"),

    # 차트 섹션
    dbc.Row([
        dbc.Col(dbc.Card(dcc.Graph(id='weekly-chart',
                figure=create_weekly_chart(df_daily))), lg=12, className="mb-4"),
        dbc.Col(dbc.Card(dcc.Graph(id='multi-year-monthly-trend',
                figure=create_multi_year_monthly_trend_chart(df_monthly.copy()))), lg=12, className="mb-4"),
        dbc.Col(dbc.Card(dcc.Graph(id='monthly-total-orders', figure=create_detailed_monthly_chart(
            df_monthly.copy(), '발주량', '월별 총 발주량'))), lg=6, className="mb-4"),
        dbc.Col(dbc.Card(dcc.Graph(id='monthly-avg-orders', figure=create_detailed_monthly_chart(
            df_monthly.copy(), '일 평균 발주량', '월별 일 평균 발주량'))), lg=6, className="mb-4"),
        dbc.Col(dbc.Card(dcc.Graph(id='monthly-bw-pages', figure=create_detailed_monthly_chart(
            df_monthly.copy(), '흑백출력량', '월별 흑백 페이지'))), lg=6, className="mb-4"),
        dbc.Col(dbc.Card(dcc.Graph(id='monthly-color-pages', figure=create_detailed_monthly_chart(
            df_monthly.copy(), '컬러 출력량', '월별 컬러 페이지'))), lg=6, className="mb-4"),
    ]),
], fluid=True)

# --- 5. 콜백 함수 (상호작용) ---


@app.callback(
    Output('kpi-cards', 'children'),
    Output('btn-this-year', 'color'),
    Output('btn-last-year', 'color'),
    Input('btn-this-year', 'n_clicks'),
    Input('btn-last-year', 'n_clicks')
)
def update_kpis(this_year_clicks, last_year_clicks):
    """요청사항 4: KPI 카드 업데이트 콜백"""
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split(
        '.')[0] if ctx.triggered else 'btn-this-year'

    current_year = datetime.now().year
    year_to_show = current_year if button_id == 'btn-this-year' else current_year - 1

    btn_this_year_color = "primary" if year_to_show == current_year else "secondary"
    btn_last_year_color = "primary" if year_to_show == current_year - 1 else "secondary"

    if df_monthly.empty:
        return dbc.Row([dbc.Col(html.P("데이터가 없습니다."), className="text-center")]), btn_this_year_color, btn_last_year_color

    year_data = df_monthly[df_monthly['월'].dt.year == year_to_show]

    if year_data.empty:
        total_orders, avg_orders, total_days = 'N/A', 'N/A', 'N/A'
    else:
        total_orders = f"{year_data['발주량'].sum():,}"
        total_days_val = year_data['발주일수'].sum()
        avg_orders = f"{(year_data['발주량'].sum() / total_days_val if total_days_val > 0 else 0):,.0f}"
        total_days = f"{total_days_val:,}"

    kpi_layout = dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader("총 발주량"), dbc.CardBody(
            html.H4(total_orders, className="card-title"))]), className="text-center"),
        dbc.Col(dbc.Card([dbc.CardHeader("일 평균 발주량"), dbc.CardBody(
            html.H4(avg_orders, className="card-title"))]), className="text-center"),
        dbc.Col(dbc.Card([dbc.CardHeader("총 발주 일수"), dbc.CardBody(
            html.H4(total_days, className="card-title"))]), className="text-center"),
    ])

    return kpi_layout, btn_this_year_color, btn_last_year_color


# --- 6. 서버 실행 ---
if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1', port=8088)

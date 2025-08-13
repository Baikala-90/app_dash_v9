import os
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from flask import Flask
from dotenv import load_dotenv

import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# gspread 및 서비스 계정 인증 관련 라이브러리 import
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# .env 파일에서 환경 변수 로드
load_dotenv()

# --- 1. Google Sheets API 설정 및 데이터 로딩 (gspread 사용) ---
SCOPES = ['https://spreadsheets.google.com/feeds',
          'https://www.googleapis.com/auth/drive']
SPREADSHEET_ID = os.getenv('SPREADSHEET_ID')
# 시트 이름 직접 명시
DAILY_SHEET_NAME = '일별 발주량 외'
MONTHLY_SHEET_NAME = '월별 발주량'


def get_google_sheets_data():
    """Google Sheets API를 통해 서비스 계정으로 인증하고 데이터를 가져옵니다."""
    try:
        # 서비스 계정 인증
        creds = ServiceAccountCredentials.from_json_keyfile_name(
            'credentials.json', SCOPES)
        client = gspread.authorize(creds)

        # SPREADSHEET_ID를 사용하여 스프레드시트 열기
        spreadsheet = client.open_by_key(SPREADSHEET_ID)

        # 시트 이름으로 워크시트 가져오기
        daily_worksheet = spreadsheet.worksheet(DAILY_SHEET_NAME)
        monthly_worksheet = spreadsheet.worksheet(MONTHLY_SHEET_NAME)

        # 모든 데이터 가져오기
        daily_values = daily_worksheet.get_all_values()
        monthly_values = monthly_worksheet.get_all_values()

        if not daily_values or not monthly_values:
            raise ValueError("시트에서 데이터를 가져오지 못했습니다. 시트가 비어있거나 데이터 범위를 확인하세요.")
        return daily_values, monthly_values

    except gspread.exceptions.SpreadsheetNotFound:
        print(f"오류: 스프레드시트를 찾을 수 없습니다. SPREADSHEET_ID가 정확한지, 서비스 계정이 시트에 공유되었는지 확인하세요.")
        return None, None
    except gspread.exceptions.WorksheetNotFound as e:
        print(
            f"오류: 워크시트를 찾을 수 없습니다. 시트 이름이 '{DAILY_SHEET_NAME}'와 '{MONTHLY_SHEET_NAME}'이 맞는지 확인하세요. ({e})")
        return None, None
    except Exception as e:
        print(f"Google Sheets 데이터 로딩 중 예상치 못한 오류 발생: {e}")
        return None, None


def process_data():
    """가져온 데이터를 Pandas DataFrame으로 변환하고 정제합니다."""
    daily_values, monthly_values = get_google_sheets_data()
    if daily_values is None or monthly_values is None:
        return None, None

    # 일별 데이터 처리 (헤더: 3행, 데이터: 4행부터)
    daily_headers = daily_values[2]
    df_daily = pd.DataFrame(daily_values[3:], columns=daily_headers)
    df_daily = df_daily[df_daily['날짜'] != ''].copy()
    df_daily['날짜'] = pd.to_datetime(df_daily['날짜'], errors='coerce')
    df_daily.dropna(subset=['날짜'], inplace=True)
    numeric_cols_daily = ['총 발주 부수', '흑백페이지', '컬러페이지']
    for col in numeric_cols_daily:
        df_daily[col] = pd.to_numeric(df_daily[col].astype(
            str).str.replace(',', ''), errors='coerce').fillna(0)

    # 월별 데이터 처리 (헤더: 2행, 데이터: 3행부터)
    monthly_headers = monthly_values[1]
    df_monthly = pd.DataFrame(monthly_values[2:], columns=monthly_headers)
    df_monthly = df_monthly[df_monthly['월'] != ''].copy()
    df_monthly['월'] = pd.to_datetime(df_monthly['월'].str.replace(
        ' ', ''), format='%Y년%m월', errors='coerce')
    df_monthly.dropna(subset=['월'], inplace=True)
    numeric_cols_monthly = ['발주량', '발주일수', '일 평균 발주량', '흑백출력량', '컬러 출력량']
    for col in numeric_cols_monthly:
        df_monthly[col] = pd.to_numeric(df_monthly[col].astype(
            str).str.replace(',', ''), errors='coerce').fillna(0)

    # 2022년 이후 데이터만 필터링
    df_monthly = df_monthly[df_monthly['월'].dt.year >= 2022].copy()
    return df_daily, df_monthly

# --- 2. 차트 생성 함수들 ---


def create_weekly_chart(df_daily):
    df_workdays = df_daily[df_daily['날짜'].dt.weekday <
                           5].sort_values('날짜', ascending=False)
    if len(df_workdays) < 10:
        return go.Figure().update_layout(title='주간 데이터가 부족합니다 (최소 10일 필요).')
    this_week_df = df_workdays.head(5).sort_values('날짜')
    last_week_df = df_workdays.iloc[5:10].sort_values('날짜')
    this_week_df['요일'] = this_week_df['날짜'].dt.strftime('%a')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=this_week_df['요일'], y=this_week_df['총 발주 부수'], mode='lines+markers',
                  name='최근 5일', text=this_week_df['날짜'].dt.strftime('%m-%d'), hoverinfo='name+y+text'))
    fig.add_trace(go.Scatter(x=this_week_df['요일'], y=last_week_df['총 발주 부수'].values, mode='lines+markers',
                  name='이전 5일', text=last_week_df['날짜'].dt.strftime('%m-%d'), hoverinfo='name+y+text', line=dict(dash='dash')))
    fig.update_layout(title='주간 발주량 비교 (최근 5영업일 vs 이전 5영업일)',
                      template='plotly_white', margin=dict(t=50, b=50, l=50, r=30))
    return fig


def create_multi_year_monthly_trend_chart(df_monthly):
    df_monthly['연도'] = df_monthly['월'].dt.year
    df_monthly['월별'] = df_monthly['월'].dt.month
    fig = go.Figure()
    years = sorted(df_monthly['연도'].unique())
    for year in years:
        year_data = df_monthly[df_monthly['연도'] == year].sort_values('월별')
        fig.add_trace(go.Scatter(
            x=year_data['월별'], y=year_data['발주량'], mode='lines+markers', name=f'{year}년'))
    fig.update_layout(title='연도별 월간 발주량 추이 (2022-현재)', xaxis_title='월', yaxis_title='발주량', template='plotly_white',
                      xaxis=dict(tickmode='array', tickvals=list(range(1, 13))), margin=dict(t=50, b=50, l=50, r=30))
    return fig


def create_detailed_monthly_chart(df_monthly, value_col, title):
    df_monthly['연도'] = df_monthly['월'].dt.year
    df_monthly['월별'] = df_monthly['월'].dt.month
    current_year = datetime.now().year
    df_pivot = df_monthly.pivot_table(
        index='월별', columns='연도', values=value_col, aggfunc='sum').fillna(0)
    fig = go.Figure()
    years_to_plot = sorted(
        [y for y in df_pivot.columns if y >= current_year - 2])
    for year in years_to_plot:
        fig.add_trace(
            go.Bar(x=df_pivot.index, y=df_pivot[year], name=f'{year}년'))
    if (current_year - 1) in df_pivot.columns and current_year in df_pivot.columns:
        last_year_vals = df_pivot[current_year - 1]
        current_year_vals = df_pivot[current_year]
        growth = ((current_year_vals - last_year_vals) /
                  last_year_vals.replace(0, pd.NA) * 100).fillna(0)
        growth_text = [f"{g:.1f}%" if g != 0 else "" for g in growth]
        fig.add_trace(go.Scatter(x=df_pivot.index, y=df_pivot[current_year], mode='text', text=growth_text, textposition='top center', textfont=dict(
            size=10, color='crimson'), showlegend=False, hoverinfo='none'))
    fig.update_layout(title=f'{title} (전년 대비 증감률)', xaxis_title='월', yaxis_title=value_col, template='plotly_white',
                      barmode='group', xaxis=dict(tickmode='array', tickvals=list(range(1, 13))), margin=dict(t=50, b=50, l=50, r=30))
    return fig


# --- 3. Dash 앱 설정 및 레이아웃 ---
server = Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[
                'https://codepen.io/chriddyp/pen/bWLwgP.css'])
app.title = "발주량 분석 대시보드"

# 데이터 로딩
df_daily, df_monthly = process_data()

# KPI 계산


def calculate_kpis(df_monthly):
    kpis = {}
    current_year = datetime.now().year
    for year in [current_year, current_year - 1]:
        year_data = df_monthly[df_monthly['월'].dt.year == year]
        if not year_data.empty:
            total_orders = year_data['발주량'].sum()
            total_days = year_data['발주일수'].sum()
            avg_orders = total_orders / total_days if total_days > 0 else 0
            kpis[year] = {'total_orders': f"{total_orders:,.0f}",
                          'avg_orders': f"{avg_orders:,.0f}", 'total_days': f"{total_days:,.0f}"}
        else:
            kpis[year] = {'total_orders': 'N/A',
                          'avg_orders': 'N/A', 'total_days': 'N/A'}
    return kpis


kpi_data = calculate_kpis(df_monthly) if df_monthly is not None else {}

# 빈 Figure 생성 함수


def create_empty_figure(message):
    fig = go.Figure()
    fig.update_layout(
        xaxis={'visible': False},
        yaxis={'visible': False},
        annotations=[{'text': message, 'xref': 'paper',
                      'yref': 'paper', 'showarrow': False, 'font': {'size': 16}}]
    )
    return fig


# 데이터 로딩 실패 시 빈 Figure, 성공 시 실제 차트 생성
if df_daily is None or df_monthly is None:
    weekly_fig = create_empty_figure("주간 데이터를 불러올 수 없습니다.")
    multi_year_fig = create_empty_figure("연간 데이터를 불러올 수 없습니다.")
    total_orders_fig = create_empty_figure("월별 총 발주량 데이터를 불러올 수 없습니다.")
    avg_orders_fig = create_empty_figure("월별 평균 발주량 데이터를 불러올 수 없습니다.")
    bw_pages_fig = create_empty_figure("흑백 페이지 데이터를 불러올 수 없습니다.")
    color_pages_fig = create_empty_figure("컬러 페이지 데이터를 불러올 수 없습니다.")
else:
    weekly_fig = create_weekly_chart(df_daily)
    multi_year_fig = create_multi_year_monthly_trend_chart(df_monthly.copy())
    total_orders_fig = create_detailed_monthly_chart(
        df_monthly.copy(), '발주량', '월별 총 발주량')
    avg_orders_fig = create_detailed_monthly_chart(
        df_monthly.copy(), '일 평균 발주량', '월별 일 평균 발주량')
    bw_pages_fig = create_detailed_monthly_chart(
        df_monthly.copy(), '흑백출력량', '월별 흑백 페이지 수')
    color_pages_fig = create_detailed_monthly_chart(
        df_monthly.copy(), '컬러 출력량', '월별 컬러 페이지 수')


app.layout = html.Div([
    html.H1('발주량 분석 대시보드', style={'textAlign': 'center'}),

    # KPI 섹션
    html.Div([
        html.H2('핵심 성과 지표 (KPI)', style={'textAlign': 'center'}),
        dcc.RadioItems(
            id='kpi-year-selector',
            options=[{'label': f'{y}년', 'value': y}
                     for y in sorted(kpi_data.keys(), reverse=True)],
            value=datetime.now().year if datetime.now().year in kpi_data else (
                datetime.now().year - 1 if (datetime.now().year - 1) in kpi_data else None),
            labelStyle={'display': 'inline-block', 'margin-right': '20px'},
            style={'textAlign': 'center', 'margin': '20px'}
        ),
        html.Div(id='kpi-cards', style={'display': 'flex',
                 'justifyContent': 'center', 'gap': '30px'})
    ], style={'marginBottom': '40px'}),

    # 차트 섹션
    html.Div([
        html.Div(dcc.Graph(id='weekly-chart', figure=weekly_fig),
                 className='six columns'),
        html.Div(dcc.Graph(id='multi-year-monthly-trend',
                 figure=multi_year_fig), className='six columns'),
    ], className='row', style={'marginBottom': '40px'}),

    html.Div([
        html.Div(dcc.Graph(id='monthly-total-orders',
                 figure=total_orders_fig), className='six columns'),
        html.Div(dcc.Graph(id='monthly-avg-orders',
                 figure=avg_orders_fig), className='six columns'),
    ], className='row', style={'marginBottom': '40px'}),

    html.Div([
        html.Div(dcc.Graph(id='monthly-bw-pages', figure=bw_pages_fig),
                 className='six columns'),
        html.Div(dcc.Graph(id='monthly-color-pages',
                 figure=color_pages_fig), className='six columns'),
    ], className='row')
])

# --- 4. Dash 콜백 ---


@app.callback(
    Output('kpi-cards', 'children'),
    Input('kpi-year-selector', 'value')
)
def update_kpi_cards(selected_year):
    if not kpi_data or selected_year not in kpi_data:
        return html.Div('KPI 데이터를 불러올 수 없습니다.', style={'textAlign': 'center', 'color': 'red', 'width': '100%'})

    data = kpi_data[selected_year]

    return [
        html.Div([html.H4('총 발주량'), html.P(data['total_orders'])], style={
                 'border': '1px solid #ddd', 'padding': '20px', 'textAlign': 'center', 'borderRadius': '5px'}),
        html.Div([html.H4('일 평균 발주량'), html.P(data['avg_orders'])], style={
                 'border': '1px solid #ddd', 'padding': '20px', 'textAlign': 'center', 'borderRadius': '5px'}),
        html.Div([html.H4('총 발주 일수'), html.P(data['total_days'])], style={
                 'border': '1px solid #ddd', 'padding': '20px', 'textAlign': 'center', 'borderRadius': '5px'}),
    ]


# --- 5. 앱 실행 ---
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8088)

# billing_page.py

import os
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from dash import dcc, html
from dotenv import load_dotenv

load_dotenv()

# --- Google Sheet 연동을 위한 공통 함수 ---


def norm(s: str) -> str:
    return re.sub(r"\s+", "", (s or "")).strip()


def find_credentials_path():
    for p in [os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
              os.getenv("GSPREAD_CREDENTIALS"),
              "credentials.json", "service_account.json"]:
        if p and os.path.exists(p):
            return p
    content = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if content:
        path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "credentials.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path
    raise FileNotFoundError("서비스 계정 JSON 파일이 없습니다.")


def open_sheet():
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    creds_file = find_credentials_path()
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_file, scope)
    client = gspread.authorize(creds)
    url = os.getenv("SPREADSHEET_URL", "").strip()
    if not url:
        raise EnvironmentError("SPREADSHEET_URL이 .env 파일에 없습니다.")
    return client.open_by_url(url)

# --- 데이터 로드 및 레이아웃 생성 ---


def load_billing_data():
    """'당월 장비 카운트 및 청구액' 시트에서 A1:J17 범위의 데이터를 불러옵니다."""
    try:
        sh = open_sheet()
        worksheet_name = os.getenv(
            "BILLING_SHEET_NAME", "당월 장비 카운트 및 청구액").strip()
        worksheet = sh.worksheet(worksheet_name)
        data = worksheet.get('A1:J17')
        return data
    except Exception as e:
        print(f"[ERROR] 청구액 데이터 로드 실패: {e}")
        return [["데이터를 불러오는 데 실패했습니다.", str(e)]]


def create_layout():
    """청구액 페이지의 전체 레이아웃을 생성합니다."""
    data = load_billing_data()

    # 데이터를 HTML 테이블로 변환
    table_header = [html.Thead(html.Tr([html.Th(cell, style={
                               'padding': '8px', 'border': '1px solid #ddd', 'textAlign': 'center', 'background': '#f2f2f2'}) for cell in data[0]]))]
    table_body = [html.Tbody([html.Tr([html.Td(cell, style={
                             'padding': '8px', 'border': '1px solid #ddd', 'textAlign': 'left'}) for cell in row]) for row in data[1:]])]

    table = html.Table(table_header + table_body,
                       style={'borderCollapse': 'collapse', 'width': '100%', 'marginTop': '20px'})

    layout = html.Div(style={'maxWidth': '1100px', 'margin': '0 auto', 'padding': '16px', 'fontFamily': 'Noto Sans KR, Malgun Gothic, Arial'}, children=[
        # H1 태그의 내용을 데이터에 맞게 동적으로 설정 (A1 셀 값 사용)
        html.H1(data[0][0] if data and data[0]
                else "당월 장비 카운트 및 청구액", style={'textAlign': 'center'}),
        dcc.Link(
            html.Button("메인 대시보드로 돌아가기", style={'marginRight': '10px'}),
            href='/'
        ),
        html.Hr(),
        html.Div(table)
    ])
    return layout

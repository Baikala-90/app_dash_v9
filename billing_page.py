# billing_page.py

from dash import dcc, html

# --- 이전에 제공해주신 '웹에 게시' URL을 정확히 반영했습니다. ---
GOOGLE_SHEET_EMBED_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSXHxDHsp-2hgmkbm-IQFmVTsOPgepYVRFr-_XIDT6yzDcE2O3vcrdnBrgv3muqUK0UTaZ-Gwz0MmeA/pubhtml?gid=906556387&single=true&widget=true&headers=false"


def create_layout():
    """청구액 페이지의 전체 레이아웃을 생성합니다."""

    layout = html.Div(style={'maxWidth': '1100px', 'margin': '0 auto', 'padding': '16px', 'fontFamily': 'Noto Sans KR, Malgun Gothic, Arial'}, children=[
        html.H1("2025 장비 카운트 및 청구액", style={'textAlign': 'center'}),
        dcc.Link(
            html.Button("메인 대시보드로 돌아가기", style={'marginRight': '10px'}),
            href='/'
        ),
        html.Hr(),

        # Iframe을 사용하여 구글 시트 표를 직접 삽입
        html.Iframe(
            src=GOOGLE_SHEET_EMBED_URL,
            style={
                'border': 'none',
                'width': '100%',
                'height': '600px'  # 표 높이에 맞게 조절 가능
            }
        )
    ])
    return layout

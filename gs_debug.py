
import os
import re
from datetime import datetime
from dotenv import load_dotenv

import gspread
from oauth2client.service_account import ServiceAccountCredentials

def guess_daily_header_row(values):
    # Look for a row whose first cell looks like a date (YYYY-MM-DD or similar)
    for i, row in enumerate(values[:15]):
        cell = (row[0] or "").strip()
        # simple date guess: contains "-" or "/" and digits
        if re.search(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', cell):
            return i
    return 3  # fallback to 3 (i.e., values[3:] in 0-index)

def guess_monthly_header_row(values):
    # Look for 'YYYY년MM월' in first cell within first 15 rows
    for i, row in enumerate(values[:15]):
        cell = (row[0] or "").replace(" ", "")
        if re.search(r'^\d{4}년\d{1,2}월$', cell):
            return i
    return 2  # fallback

def main():
    load_dotenv()
    sheet_url = os.getenv("SPREADSHEET_URL")
    print("[ENV] SPREADSHEET_URL:", sheet_url)
    if not sheet_url:
        print("!! .env 의 SPREADSHEET_URL 이 비어있습니다.")
        return

    creds_path = "credentials.json"
    if not os.path.exists(creds_path):
        print("!! credentials.json 파일이 없습니다. 프로젝트 루트에 배치해 주세요.")
        return

    # Load creds and show SA email
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
    sa_email = creds.service_account_email
    print("[SA] service account email:", sa_email)
    print("   -> 이 이메일로 해당 구글 시트를 '보기 가능' 이상 권한으로 공유했는지 확인하세요.")

    client = gspread.authorize(creds)

    # Open
    print("\n[OPEN] open_by_url 시도...")
    sh = client.open_by_url(sheet_url)
    print("[OK] Spreadsheet title:", sh.title)

    # Worksheets
    ws_list = sh.worksheets()
    print("\n[WORKSHEETS] 총", len(ws_list), "개")
    for ws in ws_list:
        print(" -", ws.title, f"(gid={ws.id})")

    # Try to find daily/monthly by common Korean titles
    candidates_daily = ["일별 발주량 외", "일별 발주량", "일별"]
    candidates_monthly = ["월별 발주량", "월별", "월별 통계"]

    def try_sheet(name_candidates):
        for name in name_candidates:
            try:
                ws = sh.worksheet(name)
                return ws
            except:
                continue
        return None

    daily_ws = try_sheet(candidates_daily)
    monthly_ws = try_sheet(candidates_monthly)

    if not daily_ws:
        print("\n!! 일별 시트를 찾지 못했습니다. 실제 탭 이름을 확인해 주세요.")
    else:
        vals = daily_ws.get_all_values()
        print(f"\n[일별] '{daily_ws.title}' 행수:", len(vals))
        if vals:
            print("  첫 3행 미리보기:")
            for r in vals[:3]:
                print("   ", r)
            hdr_idx = guess_daily_header_row(vals)
            print("  * 추정 헤더 시작 인덱스(0기준):", hdr_idx, "(즉, 데이터는 values[hdr_idx:] 로 읽어보세요)")

    if not monthly_ws:
        print("\n!! 월별 시트를 찾지 못했습니다. 실제 탭 이름을 확인해 주세요.")
    else:
        vals = monthly_ws.get_all_values()
        print(f"\n[월별] '{monthly_ws.title}' 행수:", len(vals))
        if vals:
            print("  첫 3행 미리보기:")
            for r in vals[:3]:
                print("   ", r)
            hdr_idx = guess_monthly_header_row(vals)
            print("  * 추정 헤더 시작 인덱스(0기준):", hdr_idx, "(즉, 데이터는 values[hdr_idx:] 로 읽어보세요)")

if __name__ == "__main__":
    main()


자동 시작 가이드 (Windows)

1) 필요한 파일
   - start_dashboard.bat : 앱 실행 스크립트
   - register_task.ps1   : 시작 프로그램 등록 스크립트 (작업 스케줄러)

2) 사용 방법
   a. start_dashboard.bat와 register_task.ps1를 프로젝트 폴더에 두세요:
      C:\Users\BOOKK_PRINT\발주량_대시보드

   b. PowerShell을 관리자 권한으로 실행하여 아래를 실행:
      Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
      cd "C:\Users\BOOKK_PRINT\발주량_대시보드"
      .\register_task.ps1

   c. 이후 PC를 재부팅하거나 로그아웃/로그인하면 자동으로 서버가 뜹니다.
      기본 주소: http://<서버PC-IP>:8090

3) 로그 확인
   C:\Users\BOOKK_PRINT\발주량_대시보드\logs\server.log

4) 환경
   - venv가 있으면 venv의 python을 사용합니다.
   - 없으면 C:\Users\BOOKK_PRINT\AppData\Local\Programs\Python\Python313\python.exe 또는 PATH의 python.exe를 사용합니다.
   - .env의 설정을 그대로 사용합니다. (DASH_HOST/PORT는 bat에서 재정의 가능)

5) 중지/삭제
   - 작업 스케줄러에서 "Bookk Dashboard Autostart" 비활성/삭제
   - 또는 PowerShell(관리자)에서:
     Unregister-ScheduledTask -TaskName "Bookk Dashboard Autostart" -Confirm:$false


# ====== Register Windows Scheduled Task for Bookk Dashboard ======
$TaskName = "Bookk Dashboard Autostart"
$ProjectDir = "C:\Users\BOOKK_PRINT\발주량_대시보드"
$BatPath = Join-Path $ProjectDir "start_dashboard.bat"

# Delay at startup to allow network/services to be ready
$triggerStartup = New-ScheduledTaskTrigger -AtStartup -Delay "PT20S"
$triggerLogon   = New-ScheduledTaskTrigger -AtLogOn   -Delay "PT10S"

$action = New-ScheduledTaskAction -Execute $BatPath -WorkingDirectory $ProjectDir

# Run with highest privileges (no UAC prompt in background)
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERNAME" -LogonType Interactive -RunLevel Highest

# Create or update
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger @($triggerStartup,$triggerLogon) -Principal $principal -Force

Write-Host "Registered task '$TaskName'. It will run at startup and at logon."

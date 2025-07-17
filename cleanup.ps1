# 台語語音助手專案清理腳本
# 用於清理臨時檔案、快取和舊的音檔

Write-Host "🧹 開始清理台語語音助手專案..." -ForegroundColor Green

# 清理 Python 快取
Write-Host "清理 Python 快取檔案..." -ForegroundColor Yellow
Remove-Item -Recurse -Force __pycache__ -ErrorAction SilentlyContinue
Get-ChildItem -Recurse -Name "*.pyc" | Remove-Item -Force -ErrorAction SilentlyContinue

# 清理日誌檔案
Write-Host "清理日誌檔案..." -ForegroundColor Yellow
Remove-Item app.log -ErrorAction SilentlyContinue
Remove-Item *.log -ErrorAction SilentlyContinue

# 清理重複的測試音檔
Write-Host "清理重複的測試音檔..." -ForegroundColor Yellow
$testFiles = Get-ChildItem static\ | Where-Object { 
    $_.Name -like "*li2_ho2*" -and $_.CreationTime -lt (Get-Date).AddHours(-1) 
}
if ($testFiles) {
    $testFiles | Remove-Item
    Write-Host "已刪除 $($testFiles.Count) 個重複測試音檔" -ForegroundColor Cyan
}

# 清理舊的上傳檔案（超過3天）
Write-Host "清理舊的上傳檔案..." -ForegroundColor Yellow
$oldUploads = Get-ChildItem uploads\ | Where-Object { 
    $_.LastWriteTime -lt (Get-Date).AddDays(-3) 
}
if ($oldUploads) {
    $oldUploads | Remove-Item
    Write-Host "已刪除 $($oldUploads.Count) 個舊上傳檔案" -ForegroundColor Cyan
}

# 清理舊的音檔（超過7天）
Write-Host "清理舊的音檔..." -ForegroundColor Yellow
$oldAudioFiles = Get-ChildItem static\*.wav | Where-Object { 
    $_.LastWriteTime -lt (Get-Date).AddDays(-7) 
}
if ($oldAudioFiles) {
    $oldAudioFiles | Remove-Item
    Write-Host "已刪除 $($oldAudioFiles.Count) 個舊音檔" -ForegroundColor Cyan
}

# 顯示清理後的統計
Write-Host "`n📊 清理後統計:" -ForegroundColor Green
$staticCount = (Get-ChildItem static\ | Measure-Object).Count
$uploadsCount = (Get-ChildItem uploads\ -ErrorAction SilentlyContinue | Measure-Object).Count
Write-Host "static 資料夾: $staticCount 個檔案" -ForegroundColor Cyan
Write-Host "uploads 資料夾: $uploadsCount 個檔案" -ForegroundColor Cyan

Write-Host "`n✅ 專案清理完成！" -ForegroundColor Green 
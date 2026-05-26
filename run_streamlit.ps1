# Quick start Streamlit app for Peak Accuracy ML Trading System

Write-Host "================================" -ForegroundColor Cyan
Write-Host "Peak Accuracy ML Trading App" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Check if streamlit is installed
try {
    $streamlit_check = streamlit --version
    Write-Host "✓ Streamlit installed: $streamlit_check" -ForegroundColor Green
}
catch {
    Write-Host "❌ Streamlit not found. Installing..." -ForegroundColor Yellow
    pip install streamlit -q
    Write-Host "✓ Streamlit installed" -ForegroundColor Green
}

Write-Host ""
Write-Host "Starting app..." -ForegroundColor Green
Write-Host "Browser will open at http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "Login credentials:" -ForegroundColor Yellow
Write-Host "  Admin: admin / Ahmed_Admin_2024" -ForegroundColor Gray
Write-Host "  Brother: brother / Brother_Access_2024" -ForegroundColor Gray
Write-Host ""
Write-Host "Press Ctrl+C to stop the app" -ForegroundColor Yellow
Write-Host ""

streamlit run app.py

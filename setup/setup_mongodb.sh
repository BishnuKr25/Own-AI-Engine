# PowerShell script to setup MongoDB on Windows

Write-Host "Setting up MongoDB for Sovereign AI Suite..." -ForegroundColor Green

# Create data directory
$dataPath = "C:\sovereign-ai-suite\data\mongodb"
New-Item -ItemType Directory -Force -Path $dataPath

# Create log directory
$logPath = "C:\sovereign-ai-suite\logs"
New-Item -ItemType Directory -Force -Path $logPath

# Create MongoDB config
$config = @"
systemLog:
    destination: file
    path: C:\sovereign-ai-suite\logs\mongod.log
    logAppend: true
storage:
    dbPath: C:\sovereign-ai-suite\data\mongodb
    journal:
        enabled: true
net:
    port: 27017
    bindIp: 127.0.0.1
"@

$config | Out-File -FilePath "C:\sovereign-ai-suite\config\mongod.cfg" -Encoding UTF8

Write-Host "MongoDB configuration created" -ForegroundColor Yellow

# Start MongoDB
Write-Host "Starting MongoDB..." -ForegroundColor Yellow
Start-Process "mongod" -ArgumentList "--config", "C:\sovereign-ai-suite\config\mongod.cfg" -NoNewWindow

Write-Host "MongoDB setup complete!" -ForegroundColor Green
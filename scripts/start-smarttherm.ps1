$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

function Get-MavenExecutable {
    $candidates = [System.Collections.Generic.List[string]]::new()
    $systemMaven = Get-Command mvn.cmd -ErrorAction SilentlyContinue
    if ($systemMaven) {
        $candidates.Add($systemMaven.Source)
    }

    $wrapperCache = Join-Path $HOME ".m2\wrapper\dists"
    if (Test-Path -LiteralPath $wrapperCache) {
        $cachedMavens = Get-ChildItem -LiteralPath $wrapperCache -Directory |
            Where-Object { $_.Name -match '^apache-maven-(\d+\.\d+\.\d+)$' } |
            Sort-Object { [version]($_.Name -replace '^apache-maven-', '') } -Descending
        foreach ($distribution in $cachedMavens) {
            foreach ($installation in Get-ChildItem -LiteralPath $distribution.FullName -Directory) {
                $mavenCmd = Join-Path $installation.FullName "bin\mvn.cmd"
                if (Test-Path -LiteralPath $mavenCmd) {
                    $candidates.Add($mavenCmd)
                }
            }
        }
    }

    foreach ($candidate in $candidates | Select-Object -Unique) {
        $versionOutput = (& $candidate -version 2>&1 | Out-String)
        if ($LASTEXITCODE -ne 0) {
            continue
        }
        if ($versionOutput -match 'Apache Maven\s+(\d+\.\d+\.\d+)') {
            $version = [version]$Matches[1]
            if ($version -ge [version]"3.9.0") {
                Write-Host "Using Maven $version at $candidate"
                return $candidate
            }
        }
    }

    throw "Maven 3.9+ was not found on PATH or in the standard Maven wrapper cache under '$wrapperCache'. Install Maven and add its bin folder to PATH."
}

function Assert-Command {
    param([string]$Name, [string]$Requirement)

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required tool '$Name' was not found. $Requirement"
    }
}

function Assert-PortAvailable {
    param([int]$Port)

    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
    try {
        $listener.Start()
    } catch {
        throw "Port $Port is already in use or unavailable. Close the process using 127.0.0.1:$Port and try again."
    } finally {
        $listener.Stop()
    }
}

function Wait-ForEndpoint {
    param(
        [string]$Name,
        [string]$Url,
        [int]$TimeoutSeconds
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $lastError = "No successful response was received."
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 400) {
                Write-Host "$Name is ready at $Url"
                return
            }
            $lastError = "HTTP status $($response.StatusCode)"
        } catch {
            $lastError = $_.Exception.Message
        }
        Start-Sleep -Seconds 2
    }

    throw "$Name did not become ready at $Url within $TimeoutSeconds seconds. Check its PowerShell window. Last check: $lastError"
}

Assert-Command "java" "Install Java 21 and reopen PowerShell."
Assert-Command "dotnet" "Install the .NET 10 SDK and reopen PowerShell."
Assert-Command "node" "Install Node.js 20+ and reopen PowerShell."
Assert-Command "npm.cmd" "Install Node.js 20+ and reopen PowerShell."
Assert-Command "python" "Install 64-bit Python 3.12+ and reopen PowerShell."

$javaProcessInfo = [System.Diagnostics.ProcessStartInfo]::new()
$javaProcessInfo.FileName = (Get-Command java).Source
$javaProcessInfo.Arguments = "-version"
$javaProcessInfo.UseShellExecute = $false
$javaProcessInfo.RedirectStandardOutput = $true
$javaProcessInfo.RedirectStandardError = $true
$javaProcess = [System.Diagnostics.Process]::Start($javaProcessInfo)
$javaStdout = $javaProcess.StandardOutput.ReadToEndAsync()
$javaStderr = $javaProcess.StandardError.ReadToEndAsync()
$javaProcess.WaitForExit()
$javaVersionOutput = $javaStdout.Result + $javaStderr.Result
$javaExitCode = $javaProcess.ExitCode
$javaProcess.Dispose()
if ($javaExitCode -ne 0 -or $javaVersionOutput -notmatch 'version "21(?:[.+-]|")') {
    throw "Java 21 is required. The active Java command reported: $($javaVersionOutput.Trim())"
}

$dotnetSdks = (& dotnet --list-sdks 2>&1 | Out-String)
if ($LASTEXITCODE -ne 0 -or $dotnetSdks -notmatch '(^|\r?\n)10\.') {
    throw ".NET 10 SDK is required. Install it and reopen PowerShell. Installed SDKs: $($dotnetSdks.Trim())"
}

$nodeVersionOutput = (& node --version 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $nodeVersionOutput -notmatch '^v(\d+)\.') {
    throw "Could not determine the Node.js version."
}
if ([int]$Matches[1] -lt 20) {
    throw "Node.js 20 or newer is required; found $nodeVersionOutput."
}

$pythonVersionOutput = (& python --version 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $pythonVersionOutput -notmatch '^Python\s+(\d+)\.(\d+)') {
    throw "Could not determine the Python version."
}
if ([int]$Matches[1] -lt 3 -or ([int]$Matches[1] -eq 3 -and [int]$Matches[2] -lt 12)) {
    throw "Python 3.12 or newer is required; found $pythonVersionOutput."
}

$maven = Get-MavenExecutable
foreach ($port in @(8080, 8090, 5173)) {
    Assert-PortAvailable $port
}

Write-Host "Checking the backend build and tests before starting services..."
& $maven -f (Join-Path $projectRoot "backend\pom.xml") verify
if ($LASTEXITCODE -ne 0) {
    throw "Maven verify failed with exit code $LASTEXITCODE. Review the Maven output above; no services were started."
}

$escapedRoot = $projectRoot.Replace("'", "''")
$backendJar = Join-Path $projectRoot "backend\target\smarttherm-api-1.0.0.jar"
if (-not (Test-Path -LiteralPath $backendJar)) {
    throw "The verified backend JAR was not created at '$backendJar'. Review the Maven output above."
}
$escapedBackendJar = $backendJar.Replace("'", "''")
$backendCommand = "Set-Location -LiteralPath '$escapedRoot'; & java -jar '$escapedBackendJar'"
$frontendCommand = "Set-Location -LiteralPath '$escapedRoot'; if (-not (Test-Path '.\frontend\node_modules')) { npm.cmd --prefix '.\frontend' install; if (`$LASTEXITCODE -ne 0) { exit `$LASTEXITCODE } }; npm.cmd --prefix '.\frontend' run dev -- --host 127.0.0.1"
$mlScript = Join-Path $PSScriptRoot "start-ml-service.ps1"

Start-Process -FilePath "powershell.exe" -WorkingDirectory $projectRoot -ArgumentList @("-NoExit", "-NoProfile", "-Command", $backendCommand)
Wait-ForEndpoint "Backend API" "http://127.0.0.1:8080/api/health" 90

Start-Process -FilePath "powershell.exe" -WorkingDirectory $projectRoot -ArgumentList @("-NoExit", "-NoProfile", "-Command", $frontendCommand)
Start-Process -FilePath "powershell.exe" -WorkingDirectory $projectRoot -ArgumentList @(
    "-NoExit", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $mlScript
)

Wait-ForEndpoint "Python inference service" "http://127.0.0.1:8090/health" 180
Wait-ForEndpoint "Dashboard" "http://127.0.0.1:5173/" 90
Write-Host "SmartTherm is ready. Open http://127.0.0.1:5173"

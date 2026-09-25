# Create and activate a local Python virtual environment for this project
if (-not (Test-Path .venv\Scripts\Activate.ps1)) {
	python -m venv .venv
}

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Load environment variables from .env if present
if (Test-Path .env) {
	Get-Content .env | ForEach-Object {
		$line = $_.Trim()
		if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) {
			return
		}
		$parts = $line.Split("=", 2)
		$name = $parts[0].Trim()
		$value = $parts[1].Trim().Trim('"').Trim("'")
		if ($name) {
			Set-Item -Path Env:$name -Value $value
		}
	}
	Write-Host "Loaded environment variables from .env"
} else {
	Write-Host "No .env file found. Create one from .env.template"
}

Write-Host "Virtual environment is ready. Run your scripts from this shell with .venv active."
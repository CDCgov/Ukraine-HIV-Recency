<#
Build the browser notebooks site and the Jupyter Book, and optionally serve one.

    .\jupyterlite\build_site.ps1                 # execute notebooks, build site and book
    .\jupyterlite\build_site.ps1 -SkipExecute    # build only, keep the saved outputs
    .\jupyterlite\build_site.ps1 -Serve book     # ...then open the book at http://localhost:8001
    .\jupyterlite\build_site.ps1 -Serve lab -SkipExecute -SkipBuild   # just serve the site

The build environment lives in jupyterlite\.venv (git-ignored). It is created
on first use from requirements-build.txt and requirements-notebooks.txt and is
reused afterwards. Delete the folder to rebuild it from scratch.

Precomputing the results (python jupyterlite\build_content.py) needs the full
pipeline environment and is a separate, earlier step.
#>
param(
    [switch]$SkipExecute,
    [switch]$SkipBuild,
    [ValidateSet('none', 'book', 'lab')]
    [string]$Serve = 'none',
    [int]$Port = 8001
)
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$venv = Join-Path $here '.venv'
$py = Join-Path $venv 'Scripts\python.exe'

if (-not (Test-Path $py)) {
    Write-Host "Creating the build environment in $venv ..."
    $base = (Get-Command python -ErrorAction Stop).Source
    & $base -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw 'python -m venv failed' }
    & $py -m pip install --upgrade pip
    & $py -m pip install -r (Join-Path $here 'requirements-build.txt') -r (Join-Path $here 'requirements-notebooks.txt')
    if ($LASTEXITCODE -ne 0) { throw 'pip install failed' }
}

if (-not $SkipExecute) {
    Write-Host 'Executing the notebooks ...'
    & $py (Join-Path $here 'execute_notebooks.py')
    if ($LASTEXITCODE -ne 0) { throw 'a notebook failed to execute' }
}

if (-not $SkipBuild) {
    Write-Host 'Building the JupyterLite site ...'
    Push-Location $here
    try {
        & $py -m jupyter lite build
        if ($LASTEXITCODE -ne 0) { throw 'jupyter lite build failed' }
    } finally { Pop-Location }

    Write-Host 'Building the Jupyter Book ...'
    Push-Location (Join-Path $here 'content')
    try {
        & (Join-Path $venv 'Scripts\jupyter-book.exe') build --html
        if ($LASTEXITCODE -ne 0) { throw 'jupyter book build failed' }
    } finally { Pop-Location }
}

if ($Serve -ne 'none') {
    if ($Serve -eq 'book') {
        $dir = Join-Path $here 'content\_build\html'; $url = "http://localhost:$Port/"
    } else {
        $dir = Join-Path $here '_output'; $url = "http://localhost:$Port/lab/index.html"
    }
    Write-Host "Serving $dir"
    Write-Host "Open $url (a private window avoids a stale cache). Stop with Ctrl+C."
    & $py -m http.server $Port --directory $dir
}

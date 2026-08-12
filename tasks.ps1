<#
Windows equivalent of the Makefile. Same target names.

    .\tasks.ps1 cpu-all

Exists because the primary dev box is Windows and `make` is not installed there.
The Makefile remains the authority for CI and for the Linux/Colab boxes.
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet('help', 'setup-cpu', 'lint', 'fmt', 'test-unit', 'test-contract',
                 'test-graph', 'test-integration', 'cpu-all', 'submodule', 'repro-dry',
                 'graph-smoke', 'clean')]
    [string]$Target = 'help'
)

$ErrorActionPreference = 'Stop'
$PY = if ($env:RDL_PYTHON) { $env:RDL_PYTHON } else { 'python' }

function Invoke-Step {
    param([string]$Name, [scriptblock]$Body)
    Write-Host "==> $Name" -ForegroundColor Cyan
    & $Body
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Name (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

switch ($Target) {
    'help' {
        Get-Content $PSCommandPath -TotalCount 12 | Select-Object -Skip 1
        Write-Host "targets: setup-cpu lint fmt test-unit test-contract test-integration cpu-all submodule repro-dry clean"
    }
    'setup-cpu'        { Invoke-Step 'setup-cpu' { & $PY -m pip install -e ".[cpu,dev]" } }
    'lint' {
        Invoke-Step 'ruff'  { & $PY -m ruff check src tests }
        Invoke-Step 'black' { & $PY -m black --check src tests }
        Invoke-Step 'mypy'  { & $PY -m mypy }
    }
    'fmt' {
        Invoke-Step 'ruff --fix' { & $PY -m ruff check --fix src tests }
        Invoke-Step 'black'      { & $PY -m black src tests }
    }
    'test-unit'        { Invoke-Step 'unit'        { & $PY -m pytest tests/unit } }
    'test-contract'    { Invoke-Step 'contract'    { & $PY -m pytest tests/contract } }
    # graph-unlearning-v1: the graph, defence, memory and runtime layers.
    'test-graph'       { Invoke-Step 'graph'       { & $PY -m pytest tests/graph tests/defenses tests/graph_memory tests/runtime } }
    'test-integration' { Invoke-Step 'integration' { & $PY -m pytest tests/integration } }
    'cpu-all' {
        Invoke-Step 'ruff'     { & $PY -m ruff check src tests }
        Invoke-Step 'black'    { & $PY -m black --check src tests }
        Invoke-Step 'mypy'     { & $PY -m mypy }
        Invoke-Step 'unit'     { & $PY -m pytest tests/unit }
        Invoke-Step 'contract' { & $PY -m pytest tests/contract }
        Invoke-Step 'graph'    { & $PY -m pytest tests/graph tests/defenses tests/graph_memory tests/runtime }
        Write-Host ""
        Write-Host "CPU GATE PASSED" -ForegroundColor Green
    }
    # End-to-end graph run on the stub backend, BOTH protocols. No network, no GPU,
    # nothing reportable. Mirrors the GPU sequence so a break shows up here first.
    'graph-smoke' {
        $common = @(
            '--launch', 'configs/graph/launch.yaml',
            '--fixture', 'tests/fixtures/tofu_forget10_sample.json',
            '--cohort', 'data/cohorts/graph_unlearning_v1/cpu_stub.json',
            '--limit', '4'
        )
        Invoke-Step 'graph-plan' { & $PY -m rdl.cli graph-plan @common --n-samples 2 }
        foreach ($proto in @('end_to_end_safety', 'graph_flow')) {
            $out = Join-Path $env:TEMP "rdl-graph-smoke-$proto"
            if (Test-Path $out) { Remove-Item -Recurse -Force $out }
            Invoke-Step "graph-run [$proto]" {
                & $PY -m rdl.cli graph-run @common --n-samples 2 `
                    --challenges direct --protocol $proto --output $out
            }
            Invoke-Step "graph-score [$proto]"    { & $PY -m rdl.cli graph-score --run $out }
            Invoke-Step "graph-report [$proto]"   { & $PY -m rdl.cli graph-report --run $out --challenge direct }
            Invoke-Step "graph-finalize [$proto]" { & $PY -m rdl.cli graph-finalize --run $out }
        }
        Write-Host ""
        Write-Host "GRAPH SMOKE PASSED, both protocols (diagnostic only)" -ForegroundColor Green
    }
    'submodule' { Invoke-Step 'submodule' { git submodule update --init --recursive } }
    'repro-dry' { Invoke-Step 'repro-dry' { & $PY -m rdl.cli run-repro --dry-run --condition configs/conditions/C0.yaml } }
    'clean' {
        foreach ($d in @('.pytest_cache', '.mypy_cache', '.ruff_cache', 'htmlcov')) {
            if (Test-Path $d) { Remove-Item -Recurse -Force $d }
        }
        Get-ChildItem -Recurse -Directory -Filter __pycache__ |
            ForEach-Object { Remove-Item -Recurse -Force $_.FullName }
        Write-Host "clean"
    }
}

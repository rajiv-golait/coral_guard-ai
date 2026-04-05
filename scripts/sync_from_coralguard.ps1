<#
.SYNOPSIS
  Copy ML + tabular + thesis assets from sibling CoralGuardAI/ (Colab export layout) into coralguard-api/.

.DESCRIPTION
  Source layout (mdm.py / Colab):
    CoralGuardAI/models/*.pth, *.pkl
    CoralGuardAI/data/tabular/*.pkl, *.npy
    CoralGuardAI/thesis/figures/*.png
    CoralGuardAI/image_manifest.csv -> artifacts/reference/

  Run from repo root (coralguard-api/) or pass -SourceRoot / -DestRoot.

.EXAMPLE
  .\scripts\sync_from_coralguard.ps1
  .\scripts\sync_from_coralguard.ps1 -SourceRoot "D:\Drive\MDM\CoralGuardAI"
#>
param(
    [string] $SourceRoot = "",
    [string] $DestRoot = ""
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
if (-not $DestRoot) { $DestRoot = $here }
if (-not $SourceRoot) {
    $SourceRoot = Join-Path (Split-Path -Parent $DestRoot) "CoralGuardAI"
}

if (-not (Test-Path $SourceRoot)) {
    Write-Error "Source not found: $SourceRoot. Set -SourceRoot to your CoralGuardAI folder."
}

$models = Join-Path $SourceRoot "models"
$tab = Join-Path $SourceRoot "data\tabular"
$figs = Join-Path $SourceRoot "thesis\figures"
$manifest = Join-Path $SourceRoot "image_manifest.csv"

$destModels = Join-Path $DestRoot "app\models"
$destTab = Join-Path $DestRoot "data\tabular"
$destFigs = Join-Path $DestRoot "static\thesis\figures"
$destRef = Join-Path $DestRoot "artifacts\reference"

foreach ($d in @($destModels, $destTab, $destFigs, $destRef)) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}

Write-Host "==> Models" -ForegroundColor Cyan
Copy-Item -Force (Join-Path $models "efficientnet_b3_best.pth") $destModels -ErrorAction Stop
Copy-Item -Force (Join-Path $models "dbscan_model.pkl") $destModels -ErrorAction Stop
Copy-Item -Force (Join-Path $models "ann_fusion_best.pth") $destModels -ErrorAction Stop

Write-Host "==> Tabular" -ForegroundColor Cyan
Copy-Item -Force (Join-Path $tab "features.pkl") $destTab -ErrorAction Stop
Copy-Item -Force (Join-Path $tab "scaler.pkl") $destTab -ErrorAction Stop
Copy-Item -Force (Join-Path $tab "X_train.npy") $destTab -ErrorAction Stop
$extra = @("X_val.npy", "X_test.npy", "y_train.npy", "y_val.npy", "y_test.npy")
foreach ($f in $extra) {
    $p = Join-Path $tab $f
    if (Test-Path $p) { Copy-Item -Force $p $destTab }
}

Write-Host "==> Thesis figures (PNG)" -ForegroundColor Cyan
Copy-Item -Force (Join-Path $figs "*.png") $destFigs -ErrorAction Stop

Write-Host "==> Reference manifest" -ForegroundColor Cyan
if (Test-Path $manifest) {
    Copy-Item -Force $manifest $destRef
}

Write-Host "Done. DestRoot=$DestRoot" -ForegroundColor Green

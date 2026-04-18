$nb = Get-Content -Raw "d:\KDH\NvidiaNemo\phase1_tutorial.ipynb" | ConvertFrom-Json
Write-Host "Type:" $nb.GetType().FullName
Write-Host "Top-level keys:" ($nb.PSObject.Properties.Name -join ", ")
if ($nb.PSObject.Properties.Name -contains "cells") {
	$ids = $nb.cells | Select-Object -ExpandProperty id
	Write-Host "Cell count:" $ids.Count
	$ids | Select-Object -Last 20 | ForEach-Object { Write-Host $_ }
} else {
	Write-Host "No cells property found"
}

param(
    [string]$Token,
    [string]$DatabaseId = '33507031978c81e5a8eaeaf27372f37d',
    [string]$ServerId = $env:COMPUTERNAME,
    [string]$Date = (Get-Date).ToString('yyyy-MM-dd'),
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($Token)) {
    throw 'Notion token is required. Pass -Token or set NOTION_TOKEN env and pass it in.'
}
if ([string]::IsNullOrWhiteSpace($ServerId)) {
    $ServerId = 'SERVER-UNKNOWN'
}

$headers = @{
    Authorization   = "Bearer $Token"
    'Notion-Version' = '2022-06-28'
    'Content-Type'   = 'application/json'
}

# Property IDs are used to avoid Unicode key encoding issues in some shells.
$syncDatePropId = 'iA%5EI'  # 동기화일 (date)
$serverIdPropId = 'T%7Bet'  # 서버ID (rich_text)

function Get-Title([object]$row) {
    foreach ($p in $row.properties.PSObject.Properties) {
        if ($p.Value.type -eq 'title') {
            if ($p.Value.title.Count -gt 0) {
                return (($p.Value.title | ForEach-Object { $_.plain_text }) -join '')
            }
            return ''
        }
    }
    return ''
}

$queryBody = @{ page_size = 100 } | ConvertTo-Json -Depth 10
$query = Invoke-RestMethod -Method Post -Uri "https://api.notion.com/v1/databases/$DatabaseId/query" -Headers $headers -Body $queryBody
$rows = $query.results

$targets = @()
foreach ($row in $rows) {
    $syncProp = $null
    $serverProp = $null

    foreach ($pp in $row.properties.PSObject.Properties) {
        if ($pp.Value.id -eq $syncDatePropId) { $syncProp = $pp.Value }
        if ($pp.Value.id -eq $serverIdPropId) { $serverProp = $pp.Value }
    }

    $syncMissing = $true
    $serverMissing = $true

    if ($syncProp -and $syncProp.date -and $syncProp.date.start) { $syncMissing = $false }
    if ($serverProp -and $serverProp.rich_text -and $serverProp.rich_text.Count -gt 0) { $serverMissing = $false }

    if ($syncMissing -or $serverMissing) {
        $targets += [PSCustomObject]@{
            id = $row.id
            title = Get-Title $row
            syncMissing = $syncMissing
            serverMissing = $serverMissing
        }
    }
}

Write-Output ("Found rows: {0}" -f $rows.Count)
Write-Output ("Rows needing update: {0}" -f $targets.Count)

if ($targets.Count -gt 0) {
    $targets | Select-Object -First 20 | Format-Table -AutoSize | Out-String | Write-Output
}

if ($DryRun) {
    Write-Output 'DryRun enabled. No updates applied.'
    exit 0
}

$updated = 0
$failed = 0
foreach ($t in $targets) {
    $props = @{}
    if ($t.syncMissing) {
        $props[$syncDatePropId] = @{ date = @{ start = $Date } }
    }
    if ($t.serverMissing) {
        $props[$serverIdPropId] = @{ rich_text = @(@{ type = 'text'; text = @{ content = $ServerId } }) }
    }

    $patchBody = @{ properties = $props } | ConvertTo-Json -Depth 20
    try {
        Invoke-RestMethod -Method Patch -Uri ("https://api.notion.com/v1/pages/" + $t.id) -Headers $headers -Body $patchBody | Out-Null
        $updated++
    }
    catch {
        $failed++
        if ($_.ErrorDetails -and $_.ErrorDetails.Message) {
            Write-Output ("PATCH failed: {0} :: {1}" -f $t.id, $_.ErrorDetails.Message)
        }
        else {
            Write-Output ("PATCH failed: {0} :: {1}" -f $t.id, $_.Exception.Message)
        }
    }
}

Write-Output ("updated={0} failed={1} date={2} server={3}" -f $updated, $failed, $Date, $ServerId)

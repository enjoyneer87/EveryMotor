param(
    [string]$Token,
    [string]$DatabaseId = '33507031978c81e5a8eaeaf27372f37d',
    [ValidateSet('start','heartbeat','done','hold')]
    [string]$Mode = 'heartbeat',
    [string]$TaskKey,
    [string]$ServerId = $env:EMACH_SERVER_ID,
    [string]$Note,
    [string]$CommitHash,
    [string]$RepoPath = '.',
    [string]$Date = (Get-Date).ToString('yyyy-MM-dd'),
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($Token)) {
    $Token = $env:NOTION_TOKEN
}
if ([string]::IsNullOrWhiteSpace($Token)) {
    throw 'Notion token is required. Pass -Token or set NOTION_TOKEN.'
}
if (-not [string]::IsNullOrWhiteSpace($env:NOTION_DATABASE_ID)) {
    $DatabaseId = $env:NOTION_DATABASE_ID
}
if ([string]::IsNullOrWhiteSpace($ServerId)) {
    $ServerId = $env:COMPUTERNAME
}
if ([string]::IsNullOrWhiteSpace($ServerId)) {
    $ServerId = 'SERVER-UNKNOWN'
}

$headers = @{
    Authorization    = "Bearer $Token"
    'Notion-Version' = '2022-06-28'
    'Content-Type'   = 'application/json'
}

# Property IDs (fixed for this DB)
$propStatus = 'dfGb'      # status
$propSyncDate = 'iA%5EI'  # sync date
$propServerId = 'T%7Bet'  # server id
$propCommit = 'SRSy'      # commit hash
$propVerified = 'WpGM'    # verified checkbox
$propNote = '%7BbnR'      # note
$propTaskKey = 'rC%7D%3C' # task key

# Status option IDs from DB schema
$statusInProgressId = 'bf14b907-f295-4bd6-8836-3c53f5d7fdd9'
$statusDoneId = '141570e3-a7f4-4f8b-8b88-4473c5501982'
$statusHoldId = '567a516f-af27-4d37-bab8-e8894b626ec6'

$statusIdToSet = $null
if ($Mode -eq 'start') {
    $statusIdToSet = $statusInProgressId
}
elseif ($Mode -eq 'done') {
    $statusIdToSet = $statusDoneId
}
elseif ($Mode -eq 'hold') {
    $statusIdToSet = $statusHoldId
}

if ([string]::IsNullOrWhiteSpace($CommitHash) -and $Mode -eq 'done') {
    try {
        $CommitHash = (git -C $RepoPath rev-parse --short HEAD).Trim()
    }
    catch {
        $CommitHash = ''
    }
}

function Get-RowTitle([object]$row) {
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

function Get-RichTextValue([object]$prop) {
    if (-not $prop -or -not $prop.rich_text -or $prop.rich_text.Count -eq 0) {
        return ''
    }
    return (($prop.rich_text | ForEach-Object { $_.plain_text }) -join '')
}

# Fetch all rows with pagination
$rows = @()
$cursor = $null
while ($true) {
    $queryObj = @{ page_size = 100 }
    if ($cursor) {
        $queryObj.start_cursor = $cursor
    }
    $queryBody = $queryObj | ConvertTo-Json -Depth 10

    $res = Invoke-RestMethod -Method Post -Uri "https://api.notion.com/v1/databases/$DatabaseId/query" -Headers $headers -Body $queryBody
    if ($res.results) {
        $rows += $res.results
    }

    if (-not $res.has_more) {
        break
    }
    $cursor = $res.next_cursor
}

$targets = @()
foreach ($row in $rows) {
    $pMap = @{}
    foreach ($pp in $row.properties.PSObject.Properties) {
        $pMap[$pp.Value.id] = $pp.Value
    }

    $taskKeyVal = Get-RichTextValue $pMap[$propTaskKey]
    $serverVal = Get-RichTextValue $pMap[$propServerId]
    $statusValId = ''
    if ($pMap[$propStatus] -and $pMap[$propStatus].status) {
        $statusValId = $pMap[$propStatus].status.id
    }

    $selected = $false
    if (-not [string]::IsNullOrWhiteSpace($TaskKey)) {
        if ($taskKeyVal -eq $TaskKey) {
            $selected = $true
        }
    }
    else {
        # Heartbeat default target: in-progress rows for this server or unassigned server.
        if ($statusValId -eq $statusInProgressId -and ([string]::IsNullOrWhiteSpace($serverVal) -or $serverVal -eq $ServerId)) {
            $selected = $true
        }
    }

    if ($selected) {
        $targets += [PSCustomObject]@{
            id = $row.id
            title = Get-RowTitle $row
            taskKey = $taskKeyVal
            statusId = $statusValId
        }
    }
}

Write-Output ("Mode: {0}" -f $Mode)
Write-Output ("Rows scanned: {0}" -f $rows.Count)
Write-Output ("Targets: {0}" -f $targets.Count)
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

    # Always update heartbeat metadata.
    $props[$propSyncDate] = @{ date = @{ start = $Date } }
    $props[$propServerId] = @{ rich_text = @(@{ type = 'text'; text = @{ content = $ServerId } }) }

    if ($statusIdToSet) {
        $props[$propStatus] = @{ status = @{ id = $statusIdToSet } }
    }

    if ($Mode -eq 'done') {
        $props[$propVerified] = @{ checkbox = $true }
        if (-not [string]::IsNullOrWhiteSpace($CommitHash)) {
            $props[$propCommit] = @{ rich_text = @(@{ type = 'text'; text = @{ content = $CommitHash } }) }
        }
    }
    elseif ($Mode -eq 'hold' -or $Mode -eq 'start') {
        $props[$propVerified] = @{ checkbox = $false }
    }

    if (-not [string]::IsNullOrWhiteSpace($Note)) {
        $noteText = $Note
        if ($noteText.Length -gt 1800) {
            $noteText = $noteText.Substring(0, 1800)
        }
        $props[$propNote] = @{ rich_text = @(@{ type = 'text'; text = @{ content = $noteText } }) }
    }

    $patchBody = @{ properties = $props } | ConvertTo-Json -Depth 30
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

Write-Output ("updated={0} failed={1} mode={2} date={3} server={4}" -f $updated, $failed, $Mode, $Date, $ServerId)

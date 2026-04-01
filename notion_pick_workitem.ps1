param(
    [string]$Token,
    [string]$DatabaseId = '33507031978c81e5a8eaeaf27372f37d',
    [string]$ServerId = $env:EMACH_SERVER_ID,
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

# Property IDs
$propStatus = 'dfGb'
$propSyncDate = 'iA%5EI'
$propServerId = 'T%7Bet'
$propTaskKey = 'rC%7D%3C'
$propPriority = 'Q%3AHC'

# Status option IDs
$statusStartBeforeId = 'aa66b9df-9b3c-4ed2-9e23-04067e4cc78e'
$statusInProgressId = 'bf14b907-f295-4bd6-8836-3c53f5d7fdd9'

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

function Get-PriorityScore([string]$name) {
    if ([string]::IsNullOrWhiteSpace($name)) { return 0 }
    $n = $name.ToLowerInvariant()
    if ($n -match 'p0|critical') { return 100 }
    if ($n -match 'p1|high') { return 80 }
    if ($n -match 'p2|medium') { return 60 }
    if ($n -match 'p3|low') { return 40 }
    return 50
}

# Fetch all rows
$rows = @()
$cursor = $null
while ($true) {
    $queryObj = @{ page_size = 100 }
    if ($cursor) { $queryObj.start_cursor = $cursor }
    $queryBody = $queryObj | ConvertTo-Json -Depth 10

    $res = Invoke-RestMethod -Method Post -Uri "https://api.notion.com/v1/databases/$DatabaseId/query" -Headers $headers -Body $queryBody
    if ($res.results) { $rows += $res.results }

    if (-not $res.has_more) { break }
    $cursor = $res.next_cursor
}

$inProgressMine = @()
$readyRows = @()

foreach ($row in $rows) {
    $pMap = @{}
    foreach ($pp in $row.properties.PSObject.Properties) {
        $pMap[$pp.Value.id] = $pp.Value
    }

    $statusId = ''
    if ($pMap[$propStatus] -and $pMap[$propStatus].status) {
        $statusId = $pMap[$propStatus].status.id
    }
    $serverVal = Get-RichTextValue $pMap[$propServerId]
    $taskKeyVal = Get-RichTextValue $pMap[$propTaskKey]
    $priorityName = ''
    if ($pMap[$propPriority] -and $pMap[$propPriority].select) {
        $priorityName = $pMap[$propPriority].select.name
    }

    $isTaskRow = -not [string]::IsNullOrWhiteSpace($taskKeyVal)

    if ($isTaskRow -and $statusId -eq $statusInProgressId -and $serverVal -eq $ServerId) {
        $inProgressMine += [PSCustomObject]@{
            id = $row.id
            title = Get-RowTitle $row
            taskKey = $taskKeyVal
            priority = $priorityName
            score = Get-PriorityScore $priorityName
        }
    }

    if ($isTaskRow -and $statusId -eq $statusStartBeforeId) {
        $readyRows += [PSCustomObject]@{
            id = $row.id
            title = Get-RowTitle $row
            taskKey = $taskKeyVal
            priority = $priorityName
            score = Get-PriorityScore $priorityName
        }
    }
}

Write-Output ("Rows scanned: {0}" -f $rows.Count)
Write-Output ("In-progress on this server: {0}" -f $inProgressMine.Count)

if ($inProgressMine.Count -gt 0) {
    $inProgressMine | Sort-Object -Property score -Descending | Select-Object -First 5 | Format-Table -AutoSize | Out-String | Write-Output
    Write-Output 'No reassignment required.'
    exit 0
}

if ($readyRows.Count -eq 0) {
    Write-Output 'No start-before rows available.'
    exit 0
}

$pick = $readyRows | Sort-Object -Property score -Descending | Select-Object -First 1
Write-Output 'Picked next work item:'
$pick | Format-Table -AutoSize | Out-String | Write-Output

if ($DryRun) {
    Write-Output 'DryRun enabled. No updates applied.'
    exit 0
}

$props = @{}
$props[$propStatus] = @{ status = @{ id = $statusInProgressId } }
$props[$propSyncDate] = @{ date = @{ start = $Date } }
$props[$propServerId] = @{ rich_text = @(@{ type = 'text'; text = @{ content = $ServerId } }) }

$patchBody = @{ properties = $props } | ConvertTo-Json -Depth 20
Invoke-RestMethod -Method Patch -Uri ("https://api.notion.com/v1/pages/" + $pick.id) -Headers $headers -Body $patchBody | Out-Null
Write-Output ("Assigned to server and moved to in-progress: {0}" -f $pick.id)

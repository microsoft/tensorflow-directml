# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

<#
.SYNOPSIS
Creates and emails an HTML build pipeline report.
#>
param
(
    [parameter(Mandatory)][string]$BuildId,
    [string]$AccessToken,
    [string]$EmailTo
)

if (!$AccessToken)
{
    throw "This script requires a personal access token to use the REST API."
}

$Green = '#C0FFC0'
$Red = '#FFC0C0'
$Yellow = '#FFFFAA'
$Gray = '#DDDDDD'
$LightGray = '#E6E6E6'

."$PSScriptRoot\ADOHelper.ps1"
$Ado = [ADOHelper]::CreateFromPipeline($AccessToken)
$Build = $Ado.GetBuild($BuildId)

# e.g. refs/heads/master -> master
$ShortBranchName = $Build.SourceBranch -replace '^refs/heads/'

# Only get commits associated with this build if there was at least one earlier successful build of this branch.
$FirstBuildOfBranch = $Ado.InvokeProjectApi("build/builds?definitions=$($Build.definition.id)&branchName=$($Build.SourceBranch)&`$top=1&resultFilter=succeeded&api-version=5.0", "GET", $null).Count -eq 0
if (!$FirstBuildOfBranch)
{
    $Commits = $Ado.InvokeProjectApi("build/builds/${BuildId}/changes?api-version=5.0", 'GET', $null).Value
}

$Artifacts = $Ado.InvokeProjectApi("build/builds/${BuildId}/artifacts?api-version=5.0", 'GET', $null).Value

$Html = [System.Collections.ArrayList]::new()

# Get build names from status environment variables.
$BuildNames = Get-ChildItem env:status_* | 
    Select-Object -ExpandProperty Name |
    Where-Object { $_ -match "^status_x64_(win|linux)_(release|debug)(_cp\d\d)?" } |
    ForEach-Object { ($_ -replace "^status_").ToLower() }

$SucceededBuildCount = 0
$FailedBuildCount = 0
$OtherBuildCount = 0

foreach ($BuildName in $BuildNames)
{
    # The YAML pipeline that calls this script initializes variables for each build job.
    $Status = Invoke-Expression "`$env:Status_$($BuildName -replace '-','_')"
    switch ($Status)
    {
        'Succeeded' { $SucceededBuildCount++ }
        'Failed' { $FailedBuildCount++ }
        default { $OtherBuildCount++ }
    }
}

if ($FailedBuildCount -gt 0)
{
    $BuildResult = 'Failed'
}
elseif ($SucceededBuildCount -eq $BuildNames.Count)
{
    $BuildResult = 'Succeeded'
}
else
{
    $BuildResult = 'Partially Succeeded'
}

# ---------------------------------------------------------------------------------------------------------------------
# Build Summary
# ---------------------------------------------------------------------------------------------------------------------

$BuildUrl = "https://dev.azure.com/$($Ado.Account)/$($Ado.Project)/_build/results?buildId=$($Build.ID)"
$Headers = 'Build Pipeline', 'Branch', 'Version', 'Reason', 'Duration'
$Style = "padding:1px 3px; border:1px solid gray; border-left:1px solid gray"
$Duration = (Get-Date) - [datetime]$Build.StartTime
switch ($BuildResult)
{
    'Succeeded' { $Color = $Green }
    'Failed' { $Color = $Red }
    default { $Color = $Yellow }
}

$Html += "<h1 style=`"text-align:center; border:1px solid gray; background:$Color`">Build $($BuildResult.ToUpper()) ($ShortBranchName)</h1>"
$Html += "<table style=`"border-collapse:collapse; text-align:center; width:100%;`">"
$Html += "<tr>"
foreach ($Header in $Headers)
{
    $Html += "<th style=`"text-align:center; border:1px solid gray; background-color:$LightGray; color:black;`">$Header</th>"
}
$Html += "</tr>"
$Html += '<tr>'
$Html += "<td style=`"$Style`"><a target=`"_blank`" href=`"$BuildUrl`">$($Build.BuildNumber)</a></td>"
$Html += "<td style=`"$Style`">$($Build.SourceBranch)</td>"
$Html += "<td style=`"$Style`">$($Build.SourceVersion)</td>"
$Html += "<td style=`"$Style`">$($Build.Reason)</td>"
$Html += "<td style=`"$Style`">$($Duration.ToString("c"))</td>"
$Html += '</tr>'
$Html += "</table><br>"

# ---------------------------------------------------------------------------------------------------------------------
# Build Artifacts
# ---------------------------------------------------------------------------------------------------------------------

$Headers = 'Name', 'Status', 'Agent', 'Artifacts'

$Html += "<table style=`"border-collapse:collapse; text-align:center; width:100%;`">"

$Html += "<tr>"
foreach ($Header in $Headers)
{
    $Html += "<th style=`"text-align:center; border:1px solid gray; background-color:$LightGray; color:black;`">$Header</th>"
}
$Html += "</tr>"

foreach ($BuildName in $BuildNames)
{
    # The YAML pipeline that calls this script initializes variables for each build job.
    $Status = Invoke-Expression "`$env:Status_$($BuildName -replace '-','_')"
    $Agent = Invoke-Expression "`$env:Agent_$($BuildName -replace '-','_')"

    switch ($Status)
    {
        'Succeeded' { $Color = $Green }
        'SucceededWithIssues' { $Color = $Yellow }
        'Failed' { $Color = $Red }
        default { $Status = 'Skipped'; $Color = $Gray }
    }

    $DownloadUrl = ($Artifacts | Where-Object Name -eq $BuildName).resource.downloadUrl
    $Style = "padding:1px 3px; border-bottom:1px solid gray; border-left:1px solid gray; background-color: $Color"
 
    $Html += "<tr style=`"text-align:left;`">"
    $Html += "<td style=`"$Style;`">$BuildName</td>"
    $Html += "<td style=`"$Style;`">$Status</td>"
    $Html += "<td style=`"$Style;`">$Agent</td>"
    if ($DownloadUrl)
    {
        $Html += "<td style=`"$Style; border-right:1px solid gray;`"><a href=`"$($DownloadUrl)`">$BuildName.zip</a></td>"
    }
    else
    {
        $Html += "<td style=`"$Style; border-right:1px solid gray;`">N/A</td>"
    }
    $Html += "</tr>"
}

$Html += "</table><br>"

# ---------------------------------------------------------------------------------------------------------------------
# Commits
# ---------------------------------------------------------------------------------------------------------------------

if ($Commits.Count -gt 0)
{
    $Html += "<table style=`"border-collapse:collapse; text-align:center; width:100%;`">"
    $Html += "<tr>"
    $Html += "<th style=`"text-align:center; border:1px solid gray; background-color:$LightGray; color:black;`" colspan=3>Commits</th>"
    $Html += "</tr>"

    foreach ($Commit in $Commits)
    {
        $Url = "https://dev.azure.com/$($Ado.Account)/$($Ado.Project)/_git/tensorflow/commit/$($Commit.id)"
        $Timestamp = ([datetime]$Commit.timestamp).ToString("yyyy-MM-dd HH:mm:ss")

        $Style = "padding:1px 3px; border-bottom:1px solid gray; border-left:1px solid gray"
        $Html += "<tr style=`"text-align:left;`">"
        $Html += "<td style=`"$Style; font-family:monospace;`"><a target=`"_blank`" href=`"$($Url)`">$($Commit.id.substring(0,8))</a></td>"
        $Html += "<td style=`"$Style;`">$Timestamp</td>"
        $Html += "<td style=`"$Style; border-right:1px solid gray;`"><b>$($Commit.author.uniqueName)</b> : $($Commit.message)</td>"
        $Html += "</tr>"
    }

    $Html += "</table><br>"
}

# ---------------------------------------------------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------------------------------------------------

if ($EmailTo)
{
    $EmailRecipients = @($EmailTo)
    
    $MailArgs = 
    @{
        'From'="$env:USERNAME@microsoft.com";
        'To'=$EmailRecipients;
        'Subject'="$($Build.Definition.Name) ($ShortBranchName) - $($BuildResult)";
        'Body'="$Html";
        'BodyAsHtml'=$True;
        'Priority'='Normal';
        'SmtpServer'='smtphost.redmond.corp.microsoft.com'
    }
    
    Send-MailMessage @MailArgs
}
else
{
    $Html
}
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

<#
.SYNOPSIS
Creates and emails an HTML test pipeline report.
#>
param
(
    [string]$TestArtifactsPath,
    [string[]]$BuildArtifacts,
    [string[]]$TestGroups,
    [string]$BuildRunID,
    [string]$TestRunID,
    [string]$AccessToken,
    [string]$EmailTo
)

# In case the caller passed in a single comma-separated string, such as 'dml, winml', instead of
# multiple comma-separated strings ('dml', 'winml').
$BuildArtifacts = ($BuildArtifacts -split ',').Trim()
$TestGroups = ($TestGroups -split ',').Trim()

if (!$AccessToken)
{
    throw "This script requires a personal access token to use the REST API."
}

$Green = '#C0FFC0'
$Red = '#FFC0C0'
$Yellow = '#FFFFAA'
$Gray = '#DDDDDD'
$LightGray = '#E6E6E6'

."$PSScriptRoot\..\build\ADOHelper.ps1"
$Ado = [ADOHelper]::CreateFromPipeline($AccessToken)
$Instance = $Ado.Account
$Project = $Ado.Project
$BuildRun = $Ado.GetBuild($BuildRunID)
$TestRun = $Ado.GetBuild($TestRunID)

# e.g. refs/heads/master -> master
$ShortBranchName = $BuildRun.SourceBranch -replace '^refs/heads/'

# Only get commits associated with this build if there was at least one earlier successful build of this branch.
$FirstBuildOfBranch = $Ado.InvokeProjectApi("build/builds?definitions=$($BuildRun.definition.id)&branchName=$($BuildRun.SourceBranch)&`$top=1&resultFilter=succeeded&api-version=5.0", "GET", $null).Count -eq 0
if (!$FirstBuildOfBranch)
{
    $Commits = $Ado.InvokeProjectApi("build/builds/${BuildRunID}/changes?api-version=5.0", 'GET', $null).Value
}

$DispatchInfo = Get-Content "$TestArtifactsPath/dispatch.json" -Raw | ConvertFrom-Json
$TestSummary = Get-Content "$TestArtifactsPath/test_summary.json" -Raw | ConvertFrom-Json
$TestSummaryXml = [xml](Get-Content "$TestArtifactsPath/test_summary.xml" -Raw)
$AgentSummary = Get-Content "$TestArtifactsPath/agent_summary.json" -Raw | ConvertFrom-Json

# Get test_results artifact download path
$TestResultsArtifact = $Ado.InvokeProjectApi("build/builds/$TestRunID/artifacts?artifactName=test_results&api-version=5.0", "GET", $null)
$TestResultsContainerID = $TestResultsArtifact.resource.data -replace '#/(\d+)/.*','$1'

$Html = [System.Collections.ArrayList]::new()

$TotalPassed = ($TestSummary.Groups.Agents.Counts.Pass | Measure-Object -Sum).Sum
$TotalFailed = ($TestSummary.Groups.Agents.Counts.Fail | Measure-Object -Sum).Sum
$ErrorMessages = $TestSummaryXml.assemblies.assembly.errors.error.failure.message.'#cdata-section'

if (($TotalFailed -gt 0) -or ($ErrorMessages))
{
    $TestRunResult = 'Failed'
}
elseif ($TotalPassed -eq 0)
{
    $TestRunResult = 'Skipped'
}
else
{
    $TestRunResult = 'Succeeded'
}

# ---------------------------------------------------------------------------------------------------------------------
# Tests Summary
# ---------------------------------------------------------------------------------------------------------------------

$BuildRunUrl = "https://dev.azure.com/$Instance/$Project/_build/results?buildId=$($BuildRunID)"
$TestRunUrl = "https://dev.azure.com/$Instance/$Project/_build/results?buildId=$($TestRunID)&view=ms.vss-test-web.build-test-results-tab"
$Headers = 'Build Pipeline', 'Test Pipeline', 'Branch', 'Version', 'Reason', 'Duration'
$Style = "padding:1px 3px; border:1px solid gray; border-left:1px solid gray"
$Duration = (Get-Date) - [datetime]$TestRun.StartTime
switch ($TestRunResult)
{
    'Succeeded' { $Color = $Green }
    'Failed' { $Color = $Red }
    default { $Color = $Yellow }
}

$Html += "<h1 style=`"text-align:center; border:1px solid gray; background:$Color`">Tests $($TestRunResult.ToUpper()) ($ShortBranchName)</h1>"
$Html += "<table style=`"border-collapse:collapse; text-align:center; width:100%;`">"
$Html += "<tr>"
foreach ($Header in $Headers)
{
    $Html += "<th style=`"text-align:center; border:1px solid gray; background-color:$LightGray; color:black;`">$Header</th>"
}
$Html += "</tr>"
$Html += '<tr>'
$Html += "<td style=`"$Style`"><a target=`"_blank`" href=`"$BuildRunUrl`">$($BuildRun.BuildNumber)</a></td>"
$Html += "<td style=`"$Style`"><a target=`"_blank`" href=`"$TestRunUrl`">$($TestRun.BuildNumber)</a></td>"
$Html += "<td style=`"$Style`">$($BuildRun.SourceBranch)</td>"
$Html += "<td style=`"$Style`">$($BuildRun.SourceVersion)</td>"
$Html += "<td style=`"$Style`">$($TestRun.Reason)</td>"
$Html += "<td style=`"$Style`">$($Duration.ToString("c"))</td>"
$Html += '</tr>'
$Html += "</table><br>"

# ---------------------------------------------------------------------------------------------------------------------
# Test Results
# ---------------------------------------------------------------------------------------------------------------------

$AgentNames = $DispatchInfo.AgentName | Sort-Object

$Headers = 
    'Group',
    'Agent',
    'System' ,
    'Build',
    'Errors',
    'Total',
    'Passed',
    'Failed',
    'Blocked',
    'Skipped',
    'Time'

$HeadersStyle = "border: 1px solid gray; background-color: white; color:black"
$HeaderTags = $Headers | ForEach-Object { "<th style=`"$HeadersStyle`">$_</th>" }

$Html += "<table style=`"border-collapse:collapse; text-align:center; width:100%`">"
$Html += "<tr>$HeaderTags</tr>"

foreach ($TestGroup in $TestGroups)
{
    $FirstGroupRow = $True

    $GroupResults = $TestSummary.Groups | Where-Object Name -eq $TestGroup

    # Determine group cell color.
    if     ($GroupResults.Agents.Counts.Fail -gt 0)    { $GroupColor = $Red }
    elseif ($GroupResults.Agents.Counts.Errors -gt 0)  { $GroupColor = $Red }
    elseif ($GroupResults.Agents.Counts.Blocked -gt 0) { $GroupColor = $Yellow }
    elseif ($GroupResults.Agents.Counts.Pass -gt 0)    { $GroupColor = $Green }
    else                                               { $GroupColor = $Gray }

    foreach ($AgentName in $AgentNames)
    {
        $FirstAgentRow = $True

        $AgentInfo = $AgentSummary.$AgentName
        $AgentJob = $DispatchInfo | Where-Object AgentName -eq $AgentName
        $AgentResults = $GroupResults.Agents | Where-Object Name -eq $AgentName

        if ($AgentJob.BuildID)
        {
            $AgentJobUrl = "https://$Instance.visualstudio.com/$Project/_build/results?buildId=$($AgentJob.BuildID)&view=logs"
        }
        else
        {
            $AgentJobUrl = $null
        }

        # Determine agent cell color.
        if     ($AgentResults.Counts.Fail -gt 0)    { $AgentColor = $Red }
        elseif ($AgentResults.Counts.Errors -gt 0)  { $AgentColor = $Red }
        elseif ($AgentResults.Counts.Blocked -gt 0) { $AgentColor = $Yellow }
        elseif ($AgentResults.Counts.Pass -gt 0)    { $AgentColor = $Green }
        else                                        { $AgentColor = $Gray }

        foreach ($BuildArtifact in $BuildArtifacts)
        {
            $AgentResult = $AgentResults | Where-Object Build -eq $BuildArtifact

            if ($AgentResult)
            {
                $StartTime = [datetime]$AgentResult.Start
                $EndTime = [datetime]$AgentResult.End
                $Time = New-TimeSpan $StartTime $EndTime

                # URL to download TAEF console log.
                $TaefLogURL = "https://dev.azure.com/${Instance}/_apis/resources/Containers/${TestResultsContainerID}?itemPath=test_results%2F${AgentName}%2F${BuildArtifact}%2Ftest_${TestGroup}_log.txt"
                $TaefErrorsURL = "https://dev.azure.com/${Instance}/_apis/resources/Containers/${TestResultsContainerID}?itemPath=test_results%2F${AgentName}%2F${BuildArtifact}%2Ftest_${TestGroup}_errors.txt"
            }
            else
            {
                if ($AgentJob.BuildID)
                {
                    $Time = 'NO RESULTS' # Job was assigned, but no results found.
                }
                elseif (!$AgentJob.AgentEnabled)
                {
                    $Time = 'DISABLED'
                }
                elseif ($AgentJob.AgentStatus -eq 'OFFLINE')
                {
                    $Time = 'OFFLINE'
                }
                else
                {
                    $Time = $null
                }

                $TaefLogURL = $null
            }

            $Html += "<tr>"

            # Determine result cell color.
            if     ($AgentResult.Counts.Fail -gt 0)    { $ResultColor = $Red }
            elseif ($AgentResult.Counts.Errors -gt 0)  { $ResultColor = $Red }
            elseif ($AgentResult.Counts.Blocked -gt 0) { $ResultColor = $Yellow }
            elseif ($AgentResult.Counts.Pass -gt 0)    { $ResultColor = $Green }
            else                                       { $ResultColor = $Gray }

            if ($FirstGroupRow)
            {
                $FirstGroupRow = $False
                $CellStyle = "border:1px solid gray; background-color: $GroupColor"
                $Html += "<td rowspan=`"$($AgentNames.Count * $BuildArtifacts.Count)`" colspan=`"1`" style=`"$CellStyle`">$TestGroup</td>"
            }

            $CellStyle = "border:1px solid gray; background-color: $AgentColor"
            if ($FirstAgentRow)
            {
                $FirstAgentRow = $False

                if ($AgentJobUrl)
                {
                    $Html += "<td rowspan=`"$($BuildArtifacts.Count)`" style=`"$CellStyle`"><a href=`"$AgentJobUrl`">$($AgentName)</a></td>"
                }
                else
                {
                    $Html += "<td rowspan=`"$($BuildArtifacts.Count)`" style=`"$CellStyle`">$($AgentName)</td>"
                }

                $SystemHref = "$RunPath\agent\$($AgentName)\dxdiag.xml"
                if ($AgentInfo)
                {
                    $SystemInfo = "$($AgentInfo.SystemDescription)<br>$($AgentInfo.DisplayAdapter) ($($AgentInfo.DisplayDriver))"
                }
                else
                {
                    if (!$AgentJob.AgentEnabled)
                    {
                        if ($AgentJob.IsSlowRing)
                        {
                            $SystemInfo = "Agent is disabled (slow ring)"
                        }
                        else
                        {
                            $SystemInfo = "Agent is disabled (manually)"
                        }
                    }
                    else
                    {
                        $SystemInfo = 'UNKNOWN'
                    }
                }

                $Html += "<td rowspan=`"$($BuildArtifacts.Count)`" style=`"$CellStyle; text-align: left; font-size:12px`">$SystemInfo</td>"
            }

            $CellStyle = "border:1px solid gray; background-color: $ResultColor"
    
            if ($TaefLogURL)
            {
                $Html += "<td style=`"$CellStyle`"><a href=`"$TaefLogURL`">$($BuildArtifact)</a></td>"
            }
            else
            {
                $Html += "<td style=`"$CellStyle`">$($BuildArtifact)</td>"
            }

            if ($AgentResults.Counts.Errors -gt 0)
            {
                $Html += "<td style=`"$CellStyle`"><a href=`"$TaefErrorsURL`">Yes</a></td>"
            }
            else
            {
                $Html += "<td style=`"$CellStyle`">No</td>"
            }

            $Html += "<td style=`"$CellStyle`">$($AgentResult.Counts.Total)</td>"
            $Html += "<td style=`"$CellStyle`">$($AgentResult.Counts.Pass)</td>"
            $Html += "<td style=`"$CellStyle`">$($AgentResult.Counts.Fail)</td>"
            $Html += "<td style=`"$CellStyle`">$($AgentResult.Counts.Blocked)</td>"
            $Html += "<td style=`"$CellStyle`">$($AgentResult.Counts.Skipped)</td>"
            $Html += "<td style=`"$CellStyle`">$Time</td>"
            $Html += "</tr>"
        }
    }
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
        $Url = "https://github.com/microsoft/tensorflow-directml/commit/$($Commit.id)"
        $Timestamp = ([datetime]$Commit.timestamp).ToString("yyyy-MM-dd HH:mm:ss")

        $Style = "padding:1px 3px; border-bottom:1px solid gray; border-left:1px solid gray"
        $Html += "<tr style=`"text-align:left;`">"
        $Html += "<td style=`"$Style; font-family:monospace;`"><a target=`"_blank`" href=`"$($Url)`">$($Commit.id.substring(0,8))</a></td>"
        $Html += "<td style=`"$Style;`">$Timestamp</td>"
        $Html += "<td style=`"$Style; border-right:1px solid gray;`"><b>$($Commit.author.displayName)</b> : $($Commit.message)</td>"
        $Html += "</tr>"
    }

    $Html += "</table><br>"
}

<#
# ---------------------------------------------------------------------------------------------------------------------
# Test Failure History
#
# Creates an HTML table with a row for each failing test along with up to 7 historical results:
#
# -----------------------------------------------------
# | Failing Tests  |      History (<branch>)          |
# -----------------------------------------------------
# | <test name 1>  | h1 | h2 | h3 | h4 | h5 | h6 | h7 |
# | <test name 2>  | h1 | h2 | h3 | h4 | h5 | h6 | h7 |
# | ...                      
# | <test name N>  | h1 | h2 | h3 | h4 | h5 | h6 | h7 |
# -----------------------------------------------------
#
# ... where h1-hN are either a check mark (passed), cross (failed), or dash (not run).
# ---------------------------------------------------------------------------------------------------------------------

# Get the first 50 failures from the test data.
$XmlTestResults = [xml](Get-Content "$TestArtifactsPath/test_summary.xml")
$Failures = $XmlTestResults.Assemblies.Assembly.Collection.Test | Where-Object Result -eq "Fail" | Select-Object -First 50

if ($Failures.Count -gt 0)
{
    # Add the table headers.
    $Html += "<table style=`"border-collapse:collapse; text-align:center; width:100%;`">"
    $Html += "<tr>"
    $Html += "<th style=`"text-align:center; border:1px solid gray; background-color:$LightGray; color:black;`" colspan=1>Failing Tests (First 50)</th>"
    $Html += "<th style=`"text-align:center; border:1px solid gray; background-color:$LightGray; color:black;`" colspan=7>History ($($BuildRun.SourceBranch))</th>"
    $Html += "</tr>"

    # Add a row for each failing test.
    foreach ($Failure in $Failures)
    {
        # Use REST API to get history for the current test. ADO returns only up to 7 days of history,
        # which is why the table only has 7 previous results.
        $Body = @{
            automatedTestName = $Failure.Name; 
            branch = $BuildRun.SourceBranch; 
            buildDefinitionId = $TestRun.definition.id;
            groupBy = "branch";
        } | ConvertTo-Json
        $MaxTestHistory = 7

        # Test history is nice, but not required. Failure here should not result in no email.
        try
        {
            $TestHistoryAll = $Ado.InvokeProjectApi("test/Results/testhistory?api-version=5.0-preview.1", "POST", $Body).ResultsForGroup.Results
        }
        catch
        {
            Write-Warning "Fetching test history for '$($Failure.Name)' resulted in errors! Request body = $Body"
        }
    
        # Include test results only from AP runs on the same branch. Group results from the same build, 
        # since some tests may run multiple times (e.g. in dml and metacommands group).
        $TestHistoryGroups = $TestHistoryAll | 
            Where-Object { $_.BuildReference.Number -and ($_.BuildReference.Number.EndsWith($BuildRun.SourceBranch -replace '.*/')) } | 
            Sort-Object CompletedDate | 
            Group-Object { $_.BuildReference.Number } |
            Select-Object -Last $MaxTestHistory
    
        # Choose a single result for the named test per build. $TestHistory contains up to $MaxTestHistory
        # results (passed, failed, not executed) for the current test; it may be smaller if the test is new.
        $TestHistory = [System.Collections.ArrayList]::new()
        foreach ($TestHistoryGroup in $TestHistoryGroups)
        {
            if ($TestHistoryGroup.Group.Outcome -contains "Failed")
            {
                $TestHistory += $TestHistoryGroup.Group | Where-Object Outcome -eq "Failed" | Select-Object -First 1
            }
            elseif ($TestHistoryGroup.Group.Outcome -contains "Passed")
            {
                $TestHistory += $TestHistoryGroup.Group | Where-Object Outcome -eq "Passed" | Select-Object -First 1
            }
            else
            {
                $TestHistory += $TestHistoryGroup.Group | Select-Object -First 1
            }
        }
    
        # Add column for the test name.
        $Style = "padding:1px 3px; border-bottom:1px solid gray; border-left:1px solid gray"
        $Html += "<tr style=`"text-align:left;`">"
        $Html += "<td style=`"$Style; font-family:monospace;`">$($Failure.Method)</td>"
    
        # Add columns for each test result in the history.
        for ($i = 0; $i -lt $MaxTestHistory; $i++)
        {
            $j = $i - $MaxTestHistory + $TestHistory.Count
            if ($j -ge 0)
            {
                $TestResult = $TestHistory[$j]
                $TestRunId = $TestResult.TestRun.ID
                $ResultId = $TestResult.ID
                $Outcome = $TestResult.Outcome
                $Link = "https://dev.azure.com/$Instance/$Project/_testManagement/runs?runId=$TestRunID&_a=resultSummary&resultId=$ResultID"
                $Tooltip = $TestResult.BuildReference.Number
            }
            else
            {
                $Outcome = 'NotExecuted'
                $Link = $null
                $Tooltip = $null
            }
    
            switch ($Outcome)
            {
                'Passed' { $Symbol = "&#x2714;"; $Color = $Green; }
                'Failed' { $Symbol = "&#x274C;"; $Color = $Red; }
                default { $Symbol = "-"; $Color = $LightGray; }
            }
    
            $Style = "padding:1px 3px; border-bottom:1px solid gray; border-left:1px solid gray; background:$Color;"
    
            # Add a column that shows historical test result for the failure.
            if ($Link)
            {
                $Html += "<td style=`"$Style; font-family:monospace;`"><a href=`"$Link`" alt=`"$Tooltip`">$Symbol</a></td>"
            }
            else
            {
                $Html += "<td style=`"$Style; font-family:monospace;`">$Symbol</td>"
            }
        }
    
        $Html += "</tr>"
    }
    
    $Html += "</table><br>"
}
#>

# ---------------------------------------------------------------------------------------------------------------------
# Error Messages
#
# Creates an HTML table that stores out-of-test error messages.
#
# -----------------------------------------------------
# |                   Error Messages                  |
# -----------------------------------------------------
# | <test group >  | <error message 1>                |
# | ...            | ...                              |
# | <test group >  | <error message N>                |
# -----------------------------------------------------
#
# ---------------------------------------------------------------------------------------------------------------------

if ($ErrorMessages)
{
    $Html += "<table style=`"border-collapse:collapse; text-align:center; width:100%;`">"
    $Html += "<tr><th style=`"text-align:center; border:1px solid gray; background-color:$LightGray; color:black;`" colspan=2>Error Messages</th></tr>"

    $RowCount = 0
    foreach ($Assembly in $TestSummaryXml.Assemblies.Assembly)
    {
        $AssemblyErrorMessages = $Assembly.Errors.Error.Failure.Message.'#cdata-section'
        if ($AssemblyErrorMessages)
        {
            foreach ($ErrorMessage in $AssemblyErrorMessages)
            {
                # Truncate the error message if it's too large
                if ($ErrorMessage.Length -gt 1024)
                {
                    $ErrorMessage = $ErrorMessage.Substring(0, 1021) + "..."
                }

                $AssemblyName = $Assembly.Name -replace '^autopilot::'
                $Style = "padding:1px 3px; border-bottom:1px solid gray; border-left:1px solid gray; text-align:left"
                $Html += "<tr>"
                $Html += "<td style=`"$Style; font-family:monospace;`"><b>$AssemblyName</b></td>"
                $Html += "<td style=`"$Style; font-family:monospace; border-right:1px solid gray;`">$ErrorMessage</td>"
                $Html += "</tr>"

                $RowCount++
                if ($RowCount -ge 50)
                {
                    break
                }
            }
        }
        if ($RowCount -ge 50)
        {
            break
        }
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
        'Subject'="$($TestRun.Definition.Name) ($ShortBranchName) - $($TestRunResult)";
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
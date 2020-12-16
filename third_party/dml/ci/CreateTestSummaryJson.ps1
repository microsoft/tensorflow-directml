# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

<#
.SYNOPSIS
Creates an aggregate of all test results.

.DESCRIPTION
Each tests runs on multiple environments, where an environment is defined by the agent (hardware),
build architecture, and build configuration. For example, a run may produce 4 builds 
(e.g. x64.release, x64.debug, x86.release, x86.debug) that are tested by 8 agents for up to
8*4 = 32 environments. If there are 2000 tests, then there may be up to 2000*32 = 64,000 results.

This script will summarize test results across all environments so that each test has a single state:
- Passed: test passes on all environments
- Failed: test fails on at least one environment
- Blocked: test blocks on all environments (one failure promotes the test to the failed state)
- Skipped: test skips on all environments (one failure/blocked promotes the test to failed/blocked)
#>
param
(
    [string]$TestArtifactsPath,
    [string]$OutputPath = "$TestArtifactsPath\test_summary.json"
)

# Sort test results.
$Summary = @{'Groups'=@(); 'Tests'=@{};}
$TestGroupMap = @{}
$AgentTestSummaryPaths = (Get-ChildItem "$TestArtifactsPath/*/*/test_*_summary.json").FullName
foreach ($AgentTestSummaryPath in $AgentTestSummaryPaths)
{
    Write-Host "Parsing $AgentTestSummaryPath"
    $AgentTestSummary = (Get-Content $AgentTestSummaryPath -Raw) | ConvertFrom-Json

    $GroupName = ($AgentTestSummaryPath | Split-Path -Leaf) -replace 'test_(.*)_summary.json', '$1'
    $AgentName = $AgentTestSummaryPath | Split-Path -Parent | Split-Path -Parent | Split-Path -Leaf
    $BuildName = $AgentTestSummaryPath | Split-Path -Parent | Split-Path -Leaf

    if (!$TestGroupMap[$GroupName])
    {
        $TestGroupMap[$GroupName] = @{'Name'=$GroupName; 'Agents'=@(); 'Tests'=@{}}
        $Summary.Groups += $TestGroupMap[$GroupName]
    }

    $TestGroupMap[$GroupName].Agents +=
    @{
        'Name'=$AgentName;
        'Build'=$BuildName;
        'Start'=$AgentTestSummary.Time.Start;
        'End'=$AgentTestSummary.Time.End;
        'Counts'=
        @{
            'Total'=$AgentTestSummary.Summary.Counts.Total; 
            'Pass'=$AgentTestSummary.Summary.Counts.Passed; 
            'Fail'=$AgentTestSummary.Summary.Counts.Failed; 
            'Blocked'=$AgentTestSummary.Summary.Counts.Blocked; 
            'Skipped'=$AgentTestSummary.Summary.Counts.SKipped; 
            'Errors'=if ($AgentTestSummary.Summary.Errors) {1} else {0};
        }
    }

    foreach ($TestResult in $AgentTestSummary.Summary.Tests)
    {
        # Example: [dml] DirectML.Test.UnitTests!TensorValidatorTests::ValidateNonOverlappingStrides#metadataSet546
        $Module = $TestResult.Module
        if (!$Module) { $Module = '<UnknownModule>' }
        else { $Module = $Module.Replace('.dll', '') }

        $FullTestName = "[$GroupName] $Module!$($TestResult.Name)"
        if (!$Summary.Tests[$FullTestName])
        {
            $Summary.Tests[$FullTestName] = @{
                'State'='?'
                'Fail'=@(); 
                'Pass'=@(); 
                'Blocked'=@(); 
                'Skipped'=@();
            }
        }

        $Summary.Tests[$FullTestName][$TestResult.Result] += "$AgentName!$BuildName";
    }
}

# Determine each test's state.
foreach ($TestSummary in $Summary.Tests.Values)
{
    $Total = $TestSummary.Fail.Count + $TestSummary.Pass.Count + $TestSummary.Blocked.Count + $TestSummary.Skipped.Count

    if ($TestSummary.Fail.Count -gt 0)
    {
        $TestSummary.State = 'Fail'
    }
    elseif ($TestSummary.Blocked.Count -gt 0)
    {
        $TestSummary.State = 'Blocked'
    }
    elseif ($TestSummary.Skipped.Count -eq $Total)
    {
        $TestSummary.State = 'Skipped'
    }
    else
    {
        $TestSummary.State = 'Pass'
    }
}

New-Item -ItemType File -Path $OutputPath -Force
$Summary | ConvertTo-Json -Depth 8 -Compress | Out-File $OutputPath -Encoding utf8
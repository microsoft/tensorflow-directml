# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

<#
.SYNOPSIS
Parses an AbslTest XML log to produce a flattened summary of test results.

.DESCRIPTION
This script produces an object with the following structure:

{
    "Counts":
    {
        "Passed": [int]
        "Failed": [int]
        "Skipped": [int]
        "Blocked": [int]
        "Total": [int]
    }

    "Tests":
    [
        {
            "Name": [string]
            "Result": [string] (one of "Passed", "Failed", "Skipped", or "Blocked")
            "Time" : [double] (time to execute in seconds)
            "Errors": [string] (optional; only present if result is "fail")
        }
        ...
    ]

    "Time" : [double] (time for all modules to finish, in seconds; can be smaller than the sum of test times, because of parallel execution)
    "Errors": [string] (optional; only present if errors occur outside of test scopes)
}
#>
param
(
    [Parameter(Mandatory)][string[]]$AbslTestLogPaths
)

$TestSummary = @{}
$TestSummary.Time = 0.0
$TestSummary.Tests = [Collections.ArrayList]::new()
$TestSummary.Counts = @{}
$TestSummary.Counts.Total = 0
$TestSummary.Counts.Passed = 0
$TestSummary.Counts.Failed = 0
$TestSummary.Counts.Skipped = 0
$TestSummary.Counts.Blocked = 0

try
{
    foreach ($AbslTestLogPath in $AbslTestLogPaths)
    {
        # The module name is encoded in the name of the file
        $Module = $AbslTestLogPath -replace '.*absltest_results_(.+)\.xml', '$1' | Select-Object -Unique

        $Root = ([xml](Get-Content $AbslTestLogPath -Raw)).'testsuites'

        # Since all processes start at the same time, the total execution time is the time it took for the slowest one to finish
        if ($Root.time -gt $TestSummary.Time)
        {
            $TestSummary.Time = $Root.time
        }

        foreach ($TestSuite in ($Root.testsuite))
        {
            $TestSuiteName = $TestSuite.name

            foreach ($TestCase in $TestSuite.testcase)
            {
                $TestCaseName = $TestCase.name

                $JsonTestCase = @{}
                $JsonTestCase.Name = "${TestSuiteName}.${TestCaseName}"
                $JsonTestCase.Module = $Module
                $JsonTestCase.Time = [double]::Parse($TestCase.Time)

                # Failures are saved as children nodes instead of attributes
                if ($TestCase.failure -or $TestCase.error)
                {
                    $FailureMessages = New-Object Collections.ArrayList

                    $JsonTestCase.Result = 'Fail'
                    $TestSummary.Counts.Failed++
                    $ErrorText = ''

                    if ($TestCase.failure)
                    {
                        foreach ($Failure in $TestCase.failure)
                        {
                            $FailureMessages.Add($Failure.message)
                        }
                    }

                    if ($TestCase.error)
                    {
                        foreach ($Failure in $TestCase.error)
                        {
                            $FailureMessages.Add($Failure.message)
                        }
                    }

                    foreach ($FailureMessage in $FailureMessages)
                    {
                        if ($FailureMessage -Match '.+\.(cpp|h):\d+')
                        {
                            $FilePath = $FailureMessage -Replace '(.+):\d+', '$1'
                            $LineNumber = $FailureMessage -Replace '.+:(\d+)', '$1'
                            $Message = $FailureMessage -Replace '&#xA(.+)', '$1'
                            $ErrorText += "$Message [${FilePath}:${LineNumber}]"
                        }
                        else
                        {
                            $ErrorText += $FailureMessage
                        }

                        $ErrorText = $ErrorText -Replace '&#xA', '     '
                    }

                    $JsonTestCase.Errors = $ErrorText
                }
                elseif ($TestCase.status -eq 'run' -or $TestCase.result -eq 'completed')
                {
                    # Passed GTEST tests are logged as "run" while passed abseil tests are logged as "completed"
                    $JsonTestCase.Result = 'Pass'
                    $TestSummary.Counts.Passed++
                }
                elseif ($TestCase.status -eq 'skipped' -or $TestCase.result -eq 'suppressed')
                {
                    # Skipped GTEST tests are logged as "skipped" while skipped abseil tests are logged as "suppressed"
                    $JsonTestCase.Result = 'Skipped'
                    $TestSummary.Counts.Skipped++
                }
                else
                {
                    $JsonTestCase.Result = 'Blocked'
                    $TestSummary.Counts.Blocked++
                }

                $TestSummary.Counts.Total++

                $TestSummary.Tests.Add($JsonTestCase)
            }
        }
    }
}
finally
{
    $TestSummary
}
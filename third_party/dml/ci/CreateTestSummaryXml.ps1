# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

<#
.SYNOPSIS
Converts an aggregate of all JSON test summaries into the xUnit format for VSTS.

.DESCRIPTION
This script then reads all the JSON results, builds an aggregate summary, and outputs an 
xUnit-formatted file that VSTS can use to display test results in the browser.

This script uses the following rules to aggregate the results:
- Test groups are reported as test assemblies.
- Test modules are reported as test collections.
- A test result is 'pass' if it passes on at least one agent, and has not failed on any agents.
- A test result is 'fail' if at least one agent fails the test.
- A test result is 'skipped' if all agents report either 'skipped' or 'blocked' in TAEF.
- A failing test's "errors" element contains the errors for all failing agents.
- The run date & time are the earliest run date and time from all agents.
- Test times are the median time from all agent test times.
- Assembly (test group) times are reported as the median of all agent group times.

xUnit schema: https://xunit.github.io/docs/format-xml-v2.html
#>
param
(
    [string]$TestArtifactsPath,
    [string]$OutputPath = "$TestArtifactsPath\test_summary.xml"
)

class Test
{
    $Name = 0
    $Times = @()
    $MedianTime = 0
    $Total = 0
    $Passed = 0
    $Failed = 0
    $Errors = @()
    $State = '?'

    Test($Name)
    {
        $this.Name = $Name
    }

    [void] AddResult($Test, $AgentName, $BuildName)
    {
        $this.Times += $Test.Time
        $this.Total++

        if ($Test.Errors)
        {
            $this.Errors += "[[$AgentName, $BuildName]]:`n$($Test.Errors)"
        }

        switch ($Test.Result)
        {
            'Pass' { $this.Passed++ }
            'Fail' { $this.Failed++ }
        }
    }

    [void] FinalizeResults()
    {
        if ($this.Failed -gt 0)
        {
            $this.State = 'Fail'
        }
        elseif ($this.Passed -gt 0)
        {
            $this.State = 'Pass'
        }
        else
        {
            $this.State = 'Skipped'
        }

        $this.Times = $this.Times | Sort-Object
        $this.MedianTime = $this.Times[$this.Times.Count/2]
    }

    [void] WriteXml($XmlWriter, $TestGroupName)
    {
        # In VSTS, the name of the test is used to track history, and the name of the method
        # is what actually gets displayed in the results. The 'autopilot::' prefix is used to make
        # these the aggregate test results distinct from individual agent results. In other words,
        # it will let us track the pass/fail history of the test as an aggregate. The test group
        # name is also inserted to disambiguate the history for tests that run in multiple groups.

        $XmlWriter.WriteStartElement("test")
        $XmlWriter.WriteAttributeString("name", "autopilot::$TestGroupName::$($this.Name)")
        $XmlWriter.WriteAttributeString("method", $this.Name)
        $XmlWriter.WriteAttributeString("time", $this.MedianTime)
        $XmlWriter.WriteAttributeString("result", $this.State)

        if (($this.State -ne 'Pass') -and ($this.State -ne 'Skipped'))
        {
            $XmlWriter.WriteStartElement("failure")
            $XmlWriter.WriteStartElement("message")
            $XmlWriter.WriteCData(($this.Errors -join "`n`n"))
            $XmlWriter.WriteEndElement() # message
            $XmlWriter.WriteEndElement() # failure
        }

        $XmlWriter.WriteEndElement() # test
    }
}

class TestModule
{
    $Name = ''
    $Total = 0
    $Passed = 0
    $Failed = 0
    $Skipped = 0
    $Tests = @{}

    TestModule($Name)
    {
        $this.Name = $Name
    }

    [void] AddResult($Test, $AgentName, $BuildName)
    {
        if (!$this.Tests[$Test.Name])
        {
            $this.Tests[$Test.Name] = [Test]::new($Test.Name)
        }

        $this.Tests[$Test.Name].AddResult($Test, $AgentName, $BuildName)
    }

    [void] FinalizeResults()
    {
        foreach ($Test in $this.Tests.Values)
        {
            $Test.FinalizeResults()

            switch ($Test.State)
            {
                'Pass' { $this.Passed++ }
                'Fail' { $this.Passed++ }
                'Skipped' { $this.Skipped++ }
            }
        }
    }

    [void] WriteXml($XmlWriter, $TestGroupName)
    {
        $XmlWriter.WriteStartElement("collection")
        $XmlWriter.WriteAttributeString("name", $this.Name)
        $XmlWriter.WriteAttributeString("total", $this.Total)
        $XmlWriter.WriteAttributeString("passed", $this.Passed)
        $XmlWriter.WriteAttributeString("failed", $this.Failed)
        $XmlWriter.WriteAttributeString("skipped", $this.Skipped)

        foreach ($Test in $this.Tests.Values)
        {
            $Test.WriteXml($XmlWriter, $TestGroupName)
        }

        $XmlWriter.WriteEndElement() # collection
    }
}

class TestGroup
{
    $Name = ''
    $RunDateTime = $null
    $Times = @()
    $MedianTime = 0
    $Total = 0
    $Passed = 0
    $Failed = 0
    $Skipped = 0
    $Errors = @()
    $Modules = @{}

    TestGroup($Name)
    {
        $this.Name = $Name
    }

    [void] AddSummary($SummaryPath)
    {
        $Summary = Get-Content $SummaryPath -Raw | ConvertFrom-Json
        $AgentName = $SummaryPath | Split-Path -Parent | Split-Path -Leaf
        $BuildName = $SummaryPath | Split-Path -Parent | Split-Path -Parent | Split-Path -Leaf

        $SummaryDateTime = [datetime]($Summary.Time.Start)
        if (!$this.RunDateTime -or ($SummaryDateTime -lt $this.RunDateTime))
        {
            $this.RunDateTime = $SummaryDateTime
        }

        $this.Times += $Summary.Summary.Time

        if ($Summary.Summary.Errors)
        {
            $this.Errors += "[[$AgentName, $BuildName]]:`n$($Summary.Summary.Errors)"
        }

        foreach ($Test in $Summary.Summary.Tests)
        {
            $this.AddResult($Test, $AgentName, $BuildName)
        }
    }

    [void] AddResult($Test, $AgentName, $BuildName)
    {
        $Module = $Test.Module
        if (!$Module)
        {
            $Module = 'Unknown'
        }

        if (!$this.Modules[$Module])
        {
            $this.Modules[$Module] = [TestModule]::new($Module)
        }

        $this.Modules[$Module].AddResult($Test, $AgentName, $BuildName)
    }

    [void] FinalizeResults()
    {
        foreach ($Module in $this.Modules.Values)
        {
            $Module.FinalizeResults()

            $this.Total += $Module.Total
            $this.Passed += $Module.Passed
            $this.Failed += $Module.Failed
            $this.Skipped += $Module.Skipped
        }

        $this.Times = $this.Times | Sort-Object
        if ($this.Times.Count -eq 0)
        {
            $this.MedianTime = 0
        }
        else
        {
            $this.MedianTime = $this.Times[$this.Times.Count/2]
        }
    }

    [void] WriteXml($XmlWriter)
    {
        $XmlWriter.WriteStartElement("assembly")
        $XmlWriter.WriteAttributeString("name", "autopilot::$($this.Name)")
        $XmlWriter.WriteAttributeString("test-framework", "TAEF")
        $XmlWriter.WriteAttributeString("run-date", $this.RunDateTime.ToString('yyyy-MM-dd'))
        $XmlWriter.WriteAttributeString("run-time", $this.RunDateTime.ToString('HH:mm:ss'))
        $XmlWriter.WriteAttributeString("time", $this.MedianTime)
        $XmlWriter.WriteAttributeString("total", $this.Total)
        $XmlWriter.WriteAttributeString("passed", $this.Passed)
        $XmlWriter.WriteAttributeString("failed", $this.Failed)
        $XmlWriter.WriteAttributeString("skipped", $this.Skipped)
        $XmlWriter.WriteAttributeString("errors", $this.Errors.Count)

        foreach ($Module in $this.Modules.Values)
        {
            $Module.WriteXml($XmlWriter, $this.Name)
        }

        if ($this.Errors.Count -gt 0)
        {
            $XmlWriter.WriteStartElement("errors")
            foreach ($ErrorMessage in $this.Errors)
            {
                $XmlWriter.WriteStartElement("error")
                $XmlWriter.WriteStartElement("failure")
                $XmlWriter.WriteStartElement("message")
                $XmlWriter.WriteCData($ErrorMessage)
                $XmlWriter.WriteEndElement() # message
                $XmlWriter.WriteEndElement() # failure
                $XmlWriter.WriteEndElement() # error
            }
            $XmlWriter.WriteEndElement() # errors
        }

        $XmlWriter.WriteEndElement() # assembly
    }
}


$AllSummaryFiles = (Get-ChildItem "$TestArtifactsPath\*\*\test_*_summary.json")
$Groups = $AllSummaryFiles.Name -replace 'test_(\w+)_summary.*', '$1' | Select-Object -Unique

$XmlMemoryStream = [System.IO.MemoryStream]::new()
$XmlWriterSettings = [System.Xml.XmlWriterSettings]::new()
$XmlWriterSettings.Indent = $true
$XmlWriterSettings.Encoding = [System.Text.UTF8Encoding]::new($false)
$XmlWriter = [System.Xml.XmlWriter]::Create($XmlMemoryStream, $XmlWriterSettings)
$XmlWriter.WriteStartDocument()
$XmlWriter.WriteStartElement("assemblies")

foreach ($Group in $Groups)
{
    $TestGroup = [TestGroup]::new($Group)

    $GroupSummaryFiles = $AllSummaryFiles | Where-Object Name -eq "test_${Group}_summary.json"

    foreach ($AgentSummary in $GroupSummaryFiles)
    {
        write-host "Parsing $AgentSummary"
        $TestGroup.AddSummary($AgentSummary.FullName)
    }

    Write-Host "Creating XML..."
    $TestGroup.FinalizeResults()
    $TestGroup.WriteXml($XmlWriter)
}

$XmlWriter.WriteEndElement() # assemblies
$XmlWriter.Flush()
$XmlWriter.Close()

Write-Host 'Saving XML file...'
New-Item -ItemType File -Path $OutputPath -Force
[System.Text.Encoding]::UTF8.GetString($XmlMemoryStream.ToArray()) | Out-File $OutputPath -Encoding utf8
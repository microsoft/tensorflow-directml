# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

<#
.SYNOPSIS
Dispatches an ADO pipeline job to all agents in a pool that meet certain demands.

.DESCRIPTION
This script is run on a pool controller and performs the following tasks:
- Dispatch test pipeline to all agents in the test pool
- Wait for dispatched jobs to complete
- Download test artifacts from each dispatched job
#>
Param
(
    # ID of the build that produced artifacts to test.
    [Parameter(Mandatory)][string]$BuildID,

    # Comma-separated list of build artifacts to test.
    [string]$Artifacts,

    # Comma-separated list of WSL build artifacts to test.
    [string]$WslArtifacts,

    # Comma-separated list of test groups to run on each agent.
    [string]$TestGroups,

    # Name of the test pipeline to run on agents.
    [string]$TestPipelineName = 'TF - Test Agent',

    # Name of the agent pool to receive dispatched jobs.
    [string]$TestPoolName = 'DirectML',

    # Name of the agent pool to receive dispatched jobs for WSL.
    [string]$WslTestPoolName = 'DirectML-WSL',

    # Path to store test artifacts from dispatched jobs.
    [string]$TestArtifactsPath = 'dispatch_artifacts',

    # Max time in minutes to wait for dispatched test jobs to complete.
    [int]$TimeoutMinutes = 60,

    # Personal access token to authenticate with ADO REST API.
    [parameter(Mandatory)][string]$AccessToken,

    # Ignore agents with the user capability 'AP.SlowRing' set.
    [bool]$SkipSlowRingAgents = $false
)

."$PSScriptRoot\..\build\ADOHelper.ps1"
$Ado = [ADOHelper]::CreateFromPipeline($AccessToken)

$TriggerPipeline = $Ado.GetBuild($BuildID)

$Pipeline = $Ado.GetBuildDefinition($TestPipelineName)

$AgentPool = $Ado.GetAgentPool($TestPoolName)
$AgentQueue = $Ado.GetAgentQueue($TestPoolName)
$Agents = $Ado.GetAgents($AgentPool.id)

$WslAgentPool = $Ado.GetAgentPool($WslTestPoolName)
$WslAgentQueue = $Ado.GetAgentQueue($WslTestPoolName)
$WslAgents = $Ado.GetAgents($WslAgentPool.id)

$AgentsInfo = [System.Collections.ArrayList]::new()

foreach ($Agent in $Agents)
{
    $AgentInfo = 
    @{
        'Name' = $Agent.Name;
        'Status' = $Agent.Status;
        'Enabled' = $Agent.Enabled;
        'UserCapabilities' = $Agent.UserCapabilities;
        'RunOnWsl' = 0;
        'TestPoolName' = $TestPoolName;
        'AgentQueueId' = $AgentQueue.id;
        'Artifacts' = $Artifacts;
    }

    $AgentsInfo.Add($AgentInfo)
}

foreach ($WslAgent in $WslAgents)
{
    $WslAgentInfo = 
    @{
        'Name' = $WslAgent.Name;
        'Status' = $WslAgent.Status;
        'Enabled' = $WslAgent.Enabled;
        'UserCapabilities' = $WslAgent.UserCapabilities;
        'RunOnWsl' = 1;
        'TestPoolName' = $WslTestPoolName;
        'AgentQueueId' = $WslAgentQueue.id;
        'Artifacts' = $WslArtifacts;
    }

    $AgentsInfo.Add($WslAgentInfo)
}

$DispatchedJobs = [System.Collections.ArrayList]::new()

foreach ($AgentInfo in $AgentsInfo)
{
    $JobInfo = 
    @{
        'AgentName' = $AgentInfo.Name; 
        'AgentStatus' = $AgentInfo.Status; 
        'AgentEnabled' = $AgentInfo.Enabled; 
        'IsSlowRing' = $false;
    }

    if (!$AgentInfo.Enabled)
    {
        Write-Host "Agent $($JobInfo.AgentName) is disabled. Skipping."
    }
    elseif ($AgentInfo.Status -ne 'online')
    {
        Write-Host "Agent $($JobInfo.AgentName) is offline. Skipping."
    }
    elseif ($SkipSlowRingAgents -and $AgentInfo.UserCapabilities.'AP.SlowRing')
    {
        Write-Host "Agent $($AgentInfo.Name) is in the slow ring. Skipping."
        $JobInfo.AgentEnabled = $false
        $JobInfo.IsSlowRing = $true
    }
    elseif ($AgentInfo.UserCapabilities.'AP.TargetXbox')
    {
        # TensorFlow does not run on Xbox. Skip any agents that have the AP.TargetXbox capability
        # set, which indicates a proxy agent that deploys tests to an Xbox device.
        Write-Host "Agent $($JobInfo.AgentName) is an Xbox proxy agent. Skipping."
    }
    else
    {
        $Params = 
        @{
            'Artifacts' = $AgentInfo.Artifacts;
            'TestGroups' = $TestGroups;
            'Pipeline' = $TriggerPipeline.Definition.Name;
            'PipelineBuildID' = $BuildID;
            'AgentName' = $AgentInfo.Name;
            'AgentPool' = $AgentInfo.TestPoolName;
            'TimeoutMinutes' = ($TimeoutMinutes - 5);
            'RunOnWsl' = $AgentInfo.RunOnWsl;
        }

        $Build = $Ado.QueuePipeline(
            $Pipeline.id, 
            $AgentInfo.AgentQueueId,
            $Params, 
            $env:Build_SourceBranch, 
            $env:Build_SourceVersion)

        $JobInfo.BuildID = $Build.ID

        Write-Host "Queued job $($JobInfo.BuildID) on agent $($JobInfo.AgentName)"
    }

    $DispatchedJobs += $JobInfo
}

if (!(Test-Path $TestArtifactsPath))
{
    New-Item -ItemType Directory -Path $TestArtifactsPath | Out-Null
}

$DispatchedJobs | ConvertTo-Json | Out-File "$TestArtifactsPath\dispatch.json" -Encoding utf8
$InProgressBuilds = [Collections.ArrayList]@($DispatchedJobs.BuildID)

$StartTime = Get-Date

while($InProgressBuilds -gt 0)
{
    $Builds = $Ado.GetBuilds($InProgressBuilds)
    foreach ($Build in $Builds)
    {
        if ($Build.Status -eq 'completed')
        {
            $Job = $DispatchedJobs | Where-Object { $_.BuildID -eq $Build.ID }
            Write-Host "Job completed: $($Job.AgentName)"
            $InProgressBuilds.Remove($Build.ID)

            Write-Host "Downloading test artifacts for build $($Job.BuildID)..."
            try
            {
                $OutputZip = "$TestArtifactsPath/$($Job.AgentName).zip"
                $Ado.DownloadBuildArtifacts($Build.ID, "$($Job.AgentName)", $OutputZip)
                Expand-Archive $OutputZip -DestinationPath $TestArtifactsPath
                Remove-Item $OutputZip
            }
            catch
            {
                Write-Warning "No test artifacts for job $($Job.BuildID) on agent $($Job.AgentName)!"
                Write-Warning "Error: $_"
            }
        }
    }

    if ($InProgressBuilds.Count -gt 0)
    {
        if ((New-TimeSpan $StartTime (Get-Date)).TotalMinutes -ge $TimeoutMinutes)
        {
            Write-Warning "Test jobs timed out: $InProgressBuilds"
            break
        }

        Start-Sleep -Seconds 5
    }
}
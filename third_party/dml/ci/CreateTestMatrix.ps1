# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

<#
.SYNOPSIS
Outputs a JSON-formatted matrix to test all agents in an agent pool.

.DESCRIPTION
Azure pipeline jobs can use a "matrix strategy" to spawn multiple jobs from a single job template.
The matrix is typically written inline in the job YAML, and this technique can be used to test multiple
agents with different variables; however, the drawback to this approach is that the YAML needs to be 
updated whenever an agent is added or removed. A way around this limitation is to generate the matrix 
at pipeline runtime. This script uses the REST API to scan an agent pool and create a JSON-formatted 
matrix that includes agents that are online, enabled, and have the required capabilities.

The JSON stored in the output variable is expanded, at runtime, in the pipeline job that uses it. See: 
https://docs.microsoft.com/en-us/azure/devops/pipelines/process/phases?view=azure-devops&tabs=yaml#multi-job-configuration
#>
Param
(
    # Personal access token to authenticate with ADO REST API.
    [parameter(Mandatory)][string]$AccessToken,

    # Names of the agent pool to test.
    [Parameter(Mandatory)][string[]]$AgentPoolNames,

    # List of all possible build artifacts to test.
    [Parameter(Mandatory)][string[]]$Artifacts,

    # List of all possible test groups to test.
    [Parameter(Mandatory)][string[]]$TestGroups,

    # Path to a file to store the full matrix (includes all agents in the pool).
    [Parameter(Mandatory)][string]$OutputFilePath,

    # Name of pipeline variable to store the pruned matrix (includes only agents that can be tested).
    [Parameter(Mandatory)][string]$OutputVariableName
)

."$PSScriptRoot\ADOHelper.ps1"
$Ado = [ADOHelper]::CreateFromPipeline($AccessToken)

$Matrix = @{}

foreach ($AgentPoolName in $AgentPoolNames)
{
    $AgentPool = $Ado.GetAgentPool($AgentPoolName)
    $Agents = $Ado.GetAgents($AgentPool.id)

    foreach ($Agent in $Agents)
    {
        # Flat list of "<artifact>:<testGroup>" configurations to test.
        $TestConfigurations = [System.Collections.ArrayList]::new()
    
        # Agents may support a subset of artifacts, which is stored as a regex in the 'AP.SupportedArtifacts'
        # agent capability. If not set, this will match all artifacts.
        $SupportedArtifacts = $Artifacts -match $Agent.UserCapabilities.'AP.TfArtifacts'

        foreach ($Artifact in $SupportedArtifacts)
        {
            # Agents may support a subset of test groups, which is stored as a regex in the 'AP.SupportedTestGroups'
            # agent capability. If not set, this will match all test groups.
            $SupportedTestGroups = $TestGroups -match $Agent.UserCapabilities.'AP.TfTestGroups'
    
            foreach ($TestGroup in $SupportedTestGroups)
            {
                $TestConfigurations.Add("${Artifact}:${TestGroup}") | Out-Null
            }
        }
    
        $Matrix[$Agent.Name] = [ordered]@{
            agentName = $Agent.Name;
            agentPool = $AgentPoolName;
            agentStatus = $Agent.Status;
            agentEnabled = $Agent.Enabled;
            agentTestConfigs = $TestConfigurations;
        }
    }
}

# Write the full matrix into the output file.
$Matrix.Values | ConvertTo-Json | Out-File $OutputFilePath -Encoding utf8

# Write the pruned matrix into the pipeline variable.
$PrunedMatrix = @{}
foreach ($Key in $Matrix.Keys)
{
    $Value = $Matrix[$Key]
    if ($Value.agentEnabled -and ($Value.agentStatus -eq 'online') -and $Value.agentTestConfigs)
    {
        $PrunedMatrix[$Key] = [ordered]@{
            agentName = $Value.agentName;
            agentPool = $Value.agentPool;
            agentTestConfigs = $Value.agentTestConfigs -join ',';
        }
    }
}
$PrunedMatrixJson = $PrunedMatrix | ConvertTo-Json -Compress
Write-Host "##vso[task.setVariable variable=$OutputVariableName;isOutput=true]$PrunedMatrixJson"
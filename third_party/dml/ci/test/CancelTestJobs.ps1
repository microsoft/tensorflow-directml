# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

<#
.SYNOPSIS
Cancels dispatched test jobs.

.DESCRIPTION
This script is run on a pool controller to cancel any jobs it
has previously dispatched. This ensures that test agents are not continuing
to work on jobs when the parent test pipeline is canceled.
#>
Param
(
    [string]$DispatchJsonPath,
    [string]$AccessToken
)

if (!(Test-Path $DispatchJsonPath))
{
    # If dispatch.json doesn't exist, then no jobs have been queued.
    Write-Host "No dispatch.json found."
    exit 0
}

."$PSScriptRoot\..\build\ADOHelper.ps1"
$Ado = [ADOHelper]::CreateFromPipeline($AccessToken)
$Body = @{ 'status' = 'cancelling' } | ConvertTo-Json

$DispatchedJobs = Get-Content $DispatchJsonPath -Raw | ConvertFrom-Json
foreach ($DispatchedJob in $DispatchedJobs)
{
    if ($DispatchedJob.BuildID)
    {
        Write-Host "Cancel job $($DispatchedJob.BuildID) on agent $($DispatchedJob.AgentName)"
        $Ado.InvokeProjectApi("build/builds/$($DispatchedJob.BuildID)?api-version=5.0", "PATCH", $Body)
    }
}
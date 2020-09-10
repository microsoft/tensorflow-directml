# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

param
(
    [string]$AccessToken = $env:System_AccessToken,
    [string]$ArtifactDirectory = $env:System_ArtifactsDirectory,
    [Parameter(Mandatory=$true)][string[]]$TestGroups,
    [Parameter(Mandatory=$true)][string[]]$TestArtifacts,
    [Parameter(Mandatory=$true)][string[]]$TestArtifactsBranch
)

# In case the caller passed in a single comma-separated string, such as 'models, python', instead of
# multiple comma-separated strings ('models', 'python').
$TestGroups = ($TestGroups -split ',').Trim()
$TestArtifacts = ($TestArtifacts -split ',').Trim()

."$PSScriptRoot\..\build\ADOHelper.ps1"
$Ado = [ADOHelper]::CreateFromPipeline($AccessToken)

# Get the ID of the build pipeline
$BuildPipelineID = $Ado.InvokeProjectApi("build/definitions?name=TF - Nightly Build&api-version=5.0", "GET", $null).Value.ID
if (!$BuildPipelineID)
{
    Write-Warning "Could not find build pipeline ID."
    exit 1
}

# Get the last successful build of the dml branch.
$UriParameters = @(
    "`$top=1",
    "definitions=$BuildPipelineID",
    "queryOrder=queueTimeDescending",
    "branchName=refs/heads/$TestArtifactsBranch",
    "statusFilter=completed",
    "resultFilter=succeeded",
    "api-version=5.0"
) -join '&'

$Build = $Ado.InvokeProjectApi("build/builds?$UriParameters", "GET", $null).Value
if (!$Build)
{
    Write-Warning "Could not find last successful build of $TestArtifactsBranch branch."
    exit 1
}

# Gather agent info.
Write-Host "Gathering agent info (dxdiag, etc.)..."
$TestArtifactsPath = "$ArtifactDirectory\test_results\$env:Agent_Name"
New-Item -ItemType Directory -Path $TestArtifactsPath -Force | Out-Null
Start-Process dxdiag -ArgumentList "/x $TestArtifactsPath/dxdiag.xml" -Wait
$EnvironmentVariables = @{}
Get-ChildItem "env:" | ForEach-Object { $EnvironmentVariables[$_.Name] = $_.Value }
$EnvironmentVariables | ConvertTo-Json | Out-File "$TestArtifactsPath\environment_vars.json" -Encoding utf8

# Create dispatch.json, which is needed by CreateTestEmail.ps1. Even though dispatching was done, this
# file is used to build the HTML table for agent results.
@"
[
    {
        "AgentEnabled": true,
        "AgentName": "$env:Agent_Name",
        "AgentStatus": "online"
    }
]
"@ | Out-File -FilePath "$ArtifactDirectory/test_results/dispatch.json" -Encoding utf8

foreach ($TestArtifact in $TestArtifacts)
{
    # Download and extract artifacts from the build.
    Write-Host "Downloading artifact '$TestArtifact'..."
    $BuildArtifactUrl = $Ado.InvokeProjectApi("build/builds/$($Build.ID)/artifacts?artifactName=$TestArtifact&api-version=5.0", "GET", $null).Resource.DownloadUrl
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -Uri $BuildArtifactUrl -Headers $Ado.AuthHeaders -OutFile "$ArtifactDirectory\$TestArtifact.zip"
    Write-Host "Extracting build artifacts..."

    if ($TestArtifact -like "*linux*")
    {
        $ArtifactZipPathWinAsWsl = wsl wslpath ("$ArtifactDirectory\$TestArtifact.zip" -replace '\\','/')

        $WslArtifactFolder = "/tmp/$TestArtifact"
        wsl rm -rf $WslArtifactFolder
        wsl mkdir -p /tmp
        wsl cp $ArtifactZipPathWinAsWsl /tmp/$TestArtifact.zip

        if (!(wsl command -v unzip))
        {
            Write-Host "Installing unzip..."
            wsl apt install unzip -y
        }

        if (!(wsl command -v conda))
        {
            Write-Host "Installing Miniconda3..."
            wsl wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/install_miniconda.sh
            wsl bash /tmp/install_miniconda.sh -b -p `$HOME/miniconda3
            wsl ln `$HOME/miniconda3/bin/conda /usr/bin/conda
        }

        wsl unzip /tmp/$TestArtifact.zip -d /tmp

        # This is the directory we'll be copying the test results back to at the end
        New-Item -Path $ArtifactDirectory -Name $TestArtifact -ItemType "directory"
    }
    else
    {
        Expand-Archive $ArtifactDirectory\$TestArtifact.zip -DestinationPath $ArtifactDirectory
    }

    # Run tests.
    try
    {
        Push-Location "$ArtifactDirectory\$TestArtifact"

        foreach ($TestGroup in $TestGroups)
        {
            $Results = @{'Time'=@{}}

            Write-Host "Testing $TestGroup..."
            $Results.Time.Start = (Get-Date).ToString()

            if ($TestArtifact -like "*linux*")
            {
                $WinArtifactPathAsWsl = wsl wslpath -a .

                Push-Location $WslArtifactFolderAsWin

                $LoadLibraryPath = wsl echo $WslArtifactFolder`:`$LD_LIBRARY_PATH
                $WslTensorFlowWheelPath = Get-ChildItem tensorflow_directml-*linux_x86_64.whl | Select-Object -First 1 -ExpandProperty Name
                Invoke-Expression "wsl export LD_LIBRARY_PATH='$LoadLibraryPath' '&&' python3 run_tests.py --test_group $TestGroup --tensorflow_wheel $WslTensorFlowWheelPath > test_${TestGroup}_log.txt"
                wsl find . -name '*_test_result.xml' -exec cp '{}' $WinArtifactPathAsWsl --parents \`;
                wsl find . -name '*_test_result.xml' -exec rm '{}' \`;
                wsl mv ./test_${TestGroup}_log.txt $WinArtifactPathAsWsl

                Pop-Location
            }
            else
            {
                $TensorFlowWheelPath = Get-ChildItem tensorflow_directml-*win_amd64.whl | Select-Object -First 1 -ExpandProperty Name
                py run_tests.py --test_group $TestGroup --tensorflow_wheel $TensorFlowWheelPath | Out-File -FilePath "test_${TestGroup}_log.txt"
            }

            $Results.Time.End = (Get-Date).ToString()

            $TestResultFragments = (Get-ChildItem -Path . -Filter '*_test_result.xml' -Recurse).FullName

            if ($TestResultFragments.Count -gt 0)
            {
                # We convert the AbslTest log to the same JSON format as the TAEF tests to avoid duplicating postprocessing steps.
                # After this step, there should be no differences between the 2 pipelines.
                Write-Host "Parsing $TestGroup results..."
                $TestResults = & "$PSScriptRoot\ParseAbslTestLogs.ps1" $TestResultFragments
                $TestResultFragments | Remove-Item
            }
            else
            {
                throw 'No test artifacts were produced'
            }

            $Results.Summary = $TestResults
            $Results | ConvertTo-Json -Depth 8 -Compress | Out-File "test_${TestGroup}_summary.json" -Encoding utf8
            if ($TestResults.Errors)
            {
                $TestResults.Errors | Out-File "test_${TestGroup}_errors.txt"
            }

            Write-Host "Copying $TestGroup artifacts..."
            robocopy . $TestArtifactsPath\$TestArtifact "test_${TestGroup}_*" /R:3 /W:10

            Write-Host ""
        }
    }
    finally
    {
        Pop-Location
    }
}

# Parse test results.
Write-Host "Parsing all test results..."
$AllTestArtifactsPath = $TestArtifactsPath | Split-Path -Parent
& "$PSScriptRoot/CreateTestSummaryJson.ps1" -TestArtifactsPath $AllTestArtifactsPath
& "$PSScriptRoot/CreateTestSummaryXml.ps1" -TestArtifactsPath $AllTestArtifactsPath
& "$PSScriptRoot/CreateAgentSummary.ps1" -TestArtifactsPath $AllTestArtifactsPath

# Output build ID so that future tasks in the pipeline can reference it.
Write-Output "##vso[task.setvariable variable=SrcBuildId]$($Build.Id)"

exit 0
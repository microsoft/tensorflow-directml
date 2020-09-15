# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

Param
(
    # List of test groups to run.
    [string[]]$TestGroups,

    # Path where input build artifacts are stored.
    [string]$BuildArtifactsPath,

    # Path where output test artifacts should be stored.
    [string]$TestArtifactsPath,

    # Whether we should run the tests on WSL or not
    [bool]$RunOnWsl
)

if (!(Test-Path $TestArtifactsPath))
{
    New-Item -ItemType Directory -Path $TestArtifactsPath | Out-Null
}

if ($RunOnWsl)
{
    $TestArtifact = Split-Path -Path $BuildArtifactsPath -Leaf
    $BuildArtifactPathWinAsWsl = wsl wslpath -a ($BuildArtifactsPath -replace '\\','/')
    $TestArtifactPathWinAsWsl = wsl wslpath -a ($TestArtifactsPath -replace '\\','/')
    $WslArtifactFolder = "/tmp/$TestArtifact"
    wsl rm -rf $WslArtifactFolder
    wsl mkdir -p /tmp
    wsl cp -r $BuildArtifactPathWinAsWsl $WslArtifactFolder

    $WslArtifactFolderAsWin = wsl wslpath -w $WslArtifactFolder
}

try
{
    Push-Location "$BuildArtifactsPath"

    foreach ($TestGroup in $TestGroups)
    {
        $Results = @{'Time'=@{}}

        Write-Host "Testing $TestGroup..."
        $Results.Time.Start = (Get-Date).ToString()

        if ($RunOnWsl)
        {
            Push-Location $WslArtifactFolderAsWin

            $LoadLibraryPath = wsl echo $WslArtifactFolder`:`$LD_LIBRARY_PATH
            $WslTensorFlowWheelPath = Get-ChildItem tensorflow_directml-*-linux_x86_64.whl | Select-Object -First 1 -ExpandProperty Name
            Invoke-Expression "wsl export LD_LIBRARY_PATH='$LoadLibraryPath' '&&' python3 run_tests.py --test_group $TestGroup --tensorflow_wheel $WslTensorFlowWheelPath > test_${TestGroup}_log.txt"
            wsl find . -name '*_test_result.xml' -exec cp '{}' $TestArtifactPathWinAsWsl --parents \`;
            wsl find . -name '*_test_result.xml' -exec rm '{}' \`;
            wsl mv ./test_${TestGroup}_log.txt $TestArtifactPathWinAsWsl

            Pop-Location
        }
        else
        {
            # Resolve path with wildcards
            $TensorFlowWheelPath = (Resolve-Path "$BuildArtifactsPath/tensorflow_directml-*-win_amd64.whl").Path
            py run_tests.py --test_group $TestGroup --tensorflow_wheel $TensorFlowWheelPath | Out-File -FilePath "test_${TestGroup}_log.txt"
        }

        $Results.Time.End = (Get-Date).ToString()
        $TestResultFragments = (Get-ChildItem . -Filter '*_test_result.xml' -Recurse).FullName

        if ($TestResultFragments.Count -gt 0)
        {
            # We convert the AbslTest log to the same JSON format as the TAEF tests to avoid duplicating postprocessing steps.
            # After this step, there should be no differences between the 2 pipelines.
            Write-Host "Parsing $TestGroup results..."
            $TestResults = & "$PSScriptRoot\ParseAbslTestLogs.ps1" $TestResultFragments
            $TestResultFragments | Remove-Item

            $Results.Summary = $TestResults
            $Results | ConvertTo-Json -Depth 8 -Compress | Out-File "test_${TestGroup}_summary.json" -Encoding utf8
            if ($TestResults.Errors)
            {
                $TestResults.Errors | Out-File "test_${TestGroup}_errors.txt"
            }

            Write-Host "Copying $TestGroup artifacts..."
            robocopy . $TestArtifactsPath "test_${TestGroup}_*" /R:3 /W:10
        }
        else
        {
            throw 'No test artifacts were produced'
        }

        Write-Host ""
    }
}
finally
{
    Pop-Location
}
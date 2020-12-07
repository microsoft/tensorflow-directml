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
    Write-Host "Copying the test artifacts to the WSL filesystem. This may take a while..."
    $TestArtifact = Split-Path -Path $BuildArtifactsPath -Leaf
    $BuildArtifactPathWinAsWsl = wsl wslpath -a ($BuildArtifactsPath -replace '\\','/')
    $WslArtifactFolder = "/tmp/$TestArtifact"
    wsl rm -rf $WslArtifactFolder
    wsl mkdir -p /tmp
    wsl cp -r $BuildArtifactPathWinAsWsl $WslArtifactFolder

    $WslArtifactFolderAsWin = wsl wslpath -w $WslArtifactFolder
}

foreach ($TestGroup in $TestGroups)
{
    $Results = @{'Time'=@{}}

    Write-Host "Testing $TestGroup..."
    $Results.Time.Start = (Get-Date).ToString()

    if ($RunOnWsl)
    {
        $LoadLibraryPath = wsl echo $WslArtifactFolder`:`$LD_LIBRARY_PATH
        $WslTensorFlowWheelPath = wsl ls $WslArtifactFolder/tensorflow_directml-*linux_x86_64.whl
        Invoke-Expression "wsl export LD_LIBRARY_PATH='$LoadLibraryPath' '&&' python3 $WslArtifactFolder/run_tests.py --test_group $TestGroup --tensorflow_wheel $WslTensorFlowWheelPath > $BuildArtifactsPath/test_${TestGroup}_log.txt"

        # Because of the runfiles folder, test result paths in WSL can get very long, so we give them a unique name and put them all at the root of the artifacts folder in the Windows filesystem
        wsl i=0`; for f in `$`(find $WslArtifactFolder -name *_test_result.xml`)`; do i=`$`(`(i+1`)`)`; mv `$f $BuildArtifactPathWinAsWsl/`$`{i`}_test_result.xml`; done`;
        $TestResultFragments = (Get-ChildItem $BuildArtifactsPath -Filter '*_test_result.xml').FullName
    }
    else
    {
        # Resolve path with wildcards
        $TensorFlowWheelPath = (Resolve-Path "$BuildArtifactsPath/tensorflow_directml-*-win_amd64.whl").Path
        py $BuildArtifactsPath/run_tests.py --test_group $TestGroup --tensorflow_wheel $TensorFlowWheelPath | Out-File -FilePath "$BuildArtifactsPath/test_${TestGroup}_log.txt"
        $TestResultFragments = (Get-ChildItem $BuildArtifactsPath -Filter '*_test_result.xml' -Recurse).FullName
    }

    $Results.Time.End = (Get-Date).ToString()

    if ($TestResultFragments.Count -gt 0)
    {
        # We convert the AbslTest log to the same JSON format as the TAEF tests to avoid duplicating postprocessing steps.
        # After this step, there should be no differences between the 2 pipelines.
        Write-Host "Parsing $TestGroup results..."
        $TestResults = & "$PSScriptRoot\ParseAbslTestLogs.ps1" $TestResultFragments
        $TestResultFragments | Remove-Item

        $Results.Summary = $TestResults
        $Results | ConvertTo-Json -Depth 8 -Compress | Out-File "$BuildArtifactsPath/test_${TestGroup}_summary.json" -Encoding utf8
        if ($TestResults.Errors)
        {
            $TestResults.Errors | Out-File "$BuildArtifactsPath/test_${TestGroup}_errors.txt"
        }

        Write-Host "Copying $TestGroup artifacts..."
        robocopy $BuildArtifactsPath $TestArtifactsPath "test_${TestGroup}_*" /R:3 /W:10

        # Robocopy returns non-zero exit codes for successful copies, so zero it to prevent ADO task from failing.
        if ($LASTEXITCODE -ge 8) 
        { 
            Write-Host "##[error]Robocopy failed with code $LASTEXITCODE"
        } else { $LASTEXITCODE = 0 }
    }
    else
    {
        throw 'No test artifacts were produced'
    }

    Write-Host ""
}

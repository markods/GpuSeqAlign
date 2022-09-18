# ________________________________________________________________________________________________________________
# Manual
#
# + Setup:
#    + Add Ant to the environment variable 'path' (if it isn't already there):
#       + .../NetBeans/netbeans/extide/ant/bin
#    + If the powershell won't run the script because of the execution policy, run this command:
#       + Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope CurrentUser
#    + Enjoy!
#
# + Examples:
#    + build =help
#    + build =jflex =cup =clean =build =test
#    + build =build   =compile -o codeC.obj codeC.mj   =run -debug codeC.obj
#    + build =build   =compile -o codeA.obj codeA.mj   =run -debug codeA.obj
#
# + PowerShell deep dives:
#    + https://docs.microsoft.com/en-us/powershell/scripting/learn/deep-dives/everything-about-hashtable?view=powershell-7.1
#    + https://docs.microsoft.com/en-us/powershell/scripting/learn/deep-dives/everything-about-arrays?view=powershell-7.1
#    + https://docs.microsoft.com/en-us/powershell/scripting/learn/deep-dives/everything-about-hashtable?view=powershell-7.1#saving-a-nested-hashtable-to-file
#    + https://docs.microsoft.com/en-us/powershell/scripting/learn/ps101/06-flow-control?view=powershell-7.1
#    + https://docs.microsoft.com/en-us/powershell/scripting/learn/ps101/09-functions?view=powershell-7.1
#
# + Specific stuff:
#    + https://docs.microsoft.com/en-us/dotnet/api/system.collections.arraylist?view=net-5.0
#    + https://powershellexplained.com/2017-11-20-Powershell-stringBuilder/
#    + https://docs.microsoft.com/en-us/powershell/scripting/learn/deep-dives/everything-about-null?view=powershell-7.1
#    + https://docs.microsoft.com/en-us/powershell/module/microsoft.powershell.management/remove-item?view=powershell-7.1#example-4--delete-files-in-subfolders-recursively
#    + https://docs.microsoft.com/en-us/powershell/scripting/learn/deep-dives/everything-about-switch?view=powershell-7.1
#    + https://docs.microsoft.com/en-us/powershell/scripting/learn/deep-dives/everything-about-hashtable?view=powershell-7.1#splatting-hashtables-at-cmdlets
#    + https://docs.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_operators?view=powershell-7.1
#
# + Important:
#    + https://docs.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_classes?view=powershell-7.2#output-in-class-methods
#    + https://github.com/PowerShell/PowerShell/issues/4616

using namespace System.Collections.Generic;
using namespace System.Management.Automation;
using namespace System.Collections.Specialized;
using namespace System.Text.Json;





# ________________________________________________________________________________________________________________
# Framework

[string] $script:ProjectRoot = Split-Path -Path $MyInvocation.MyCommand.Path -Parent;
[string] $script:Target = 'nw.exe';   # compilation target
[int] $script:LastStatusCode = 0;

[string] $script:StageSep = "------------------------------------------------------------------------------------------------------ <<< {0}";
[string] $script:LineSep  = "------------------";

# NOTE: leave powershell array constructor ( @() ) if there is only one argument (otherwise it won't be a powershell array due to unpacking)
[string[][]] $script:DefaultArgs =
    ( "=clean", "=build", "=run", "subst1-blosum.json", "param1-best.json", "seq3-gen-10k.json" ),

    (                     "=run", "subst1-blosum.json", "param0-test.json", "seq0-test.json"    ),

    (                     "=run", "subst1-blosum.json", "param2-all.json",  "seq1-gen-0.1k.json"),
    (                     "=run", "subst1-blosum.json", "param2-all.json",  "seq1-gen-1k.json"  ),
    (                     "=run", "subst1-blosum.json", "param2-all.json",  "seq1-gen-10k.json" ),

    (                     "=run", "subst1-blosum.json", "param1-best.json", "seq1-gen-0.1k.json"),
    (                     "=run", "subst1-blosum.json", "param1-best.json", "seq1-gen-1k.json"  ),
    (                     "=run", "subst1-blosum.json", "param1-best.json", "seq1-gen-10k.json" );

[string] $script:HelpMessage = @"
build   [[-]-help]   [=clean ...] [=build ...]   [=run fsubsts fparams fseqs ...\n]

Default:             build   --help

Switches:
    -help            show the help menu
    -0               use the default parameters 0:   build-v2 $( $script:DefaultArgs[ 0 ] -Join ' ')
    
    -1               use the default parameters 1:   build-v2 $( $script:DefaultArgs[ 1 ] -Join ' ')

    -2               use the default parameters 2:   build-v2 $( $script:DefaultArgs[ 2 ] -Join ' ')
    -3               use the default parameters 3:   build-v2 $( $script:DefaultArgs[ 3 ] -Join ' ')
    -4               use the default parameters 4:   build-v2 $( $script:DefaultArgs[ 4 ] -Join ' ')

Switches:
    --help           shows the help menu
    -help            same as --help

    =clean           clean project
       -all          +   clean logs and .vs folder as well
    =build           build project
       -debug        +   don't optimise and define the DEBUG symbol

    =run             run the compiled program
       fsubsts       +   specify the "substitution matrices" json file (relative to /resrc)
       fparams       +   specify the "nw algorithm parameters" json file (relative to /resrc)
       fseqs         +   specify the "sequences to be compared" json file (relative to /resrc)

"@;

[scriptblock] $script:StageScript_Default =
{
    param( [Stage] $Stage, [bool] $PrintStageName = $true )

    if( $Stage.CmdPartArr.Count -eq 0 ) { $script:LastStatusCode = -1; return; }

    # print the stage name if requested
    if( $PrintStageName ) { $script:StageSep -f $Stage.Name | Write-Output; }
    # print the stage command
    $Command = $Stage.GetCommand();
    $Command | Write-Output;

    # if the subcommand doesn't accept arguments but they were given anyway (if the subcommand is simple)
    # IMPORTANT: && and || are pipeline chain operators!, not logical operators (-and and -or)
    $CmdArgArr = $Stage.CmdArgArr;
    if( !$Stage.AcceptsArgs   -and   $CmdArgArr.Count -gt 0 )
    {
        "Subcommand does not accept arguments" | Write-Output;
        $script:LastStatusCode = 400; return;
    }

    # invoke the stage's command
    Invoke-Expression -Command $Command;
    # if the command invocation failed, return
    if( $? -ne $true )
    {
        "Subcommand invocation failure" | Write-Output;
        $script:LastStatusCode = 400; return;
    }
    # if an error occured in the command, return
    if( $LASTEXITCODE -ne 0 )
    {
        "Subcommand invocation failure" | Write-Output;
        $script:LastStatusCode = $LASTEXITCODE; return;
    }
    
    $script:LastStatusCode = 0; return;
}



# class FileUtil
# {
    # find all files with the given extension recursively relative to the given location
    function FileUtil_FindRecursive
    {
        param( [string] $Location, [string] $Extension )

        # always initialize variables before use (since an exception can occur during assignment)
        $Files = $null;
        # find files with the given extension recursively in the current working directory
        $Files = Get-ChildItem -Path "$PWD" -Include "*$Extension" -File -Recurse;

        # for all files, get the full name from the file info, get the path relative to the current working directory, and finally convert the result to a string
        # +   return an array of such files
        $Result = @();
        if( $Files )
        {
            try
            {
                Push-Location $Location;
                $Result = @( $Files.FullName | Resolve-Path -Relative | ForEach-Object -Process { $_.ToString() } );
            }
            catch
            {
                "Could not find all items with the extension '{0}' recursively on the path '{1}'" -f $Extension, $Path | Write-Output;
                $script:LastStatusCode = -1;
            }
            finally
            {
                Pop-Location;
            }
        }

        return $Result;
    }

    function FileUtil_MoveItem
    {
        param( [string] $Path, [string] $Destination )

        [bool] $PathExists = Test-Path $Path -PathType "Any";
        if( $? -ne $true )
        {
            "Could not test if item exists: '{0}'" -f $Path | Write-Output;
            $script:LastStatusCode = -1; return;
        }
        if( !$PathExists ) { $script:LastStatusCode = 0; return; }

        Move-Item -Path $Path -Destination $Destination *>&1 | Out-Null;
        if( $? -ne $true )
        {
            "Could not move item: '{0}'" -f $Path | Write-Output;
            $script:LastStatusCode = -1; return ;
        }

        $script:LastStatusCode = 0; return ;
    }

    function FileUtil_RemoveFolder
    {
        param( [string] $Path )

        [bool] $PathExists = Test-Path $Path -PathType "Container";
        if( $? -ne $true )
        {
            "Could not test if folder exists: '{0}'" -f $Path | Write-Output;
            $script:LastStatusCode = -1; return;
        }
        if( !$PathExists ) { $script:LastStatusCode = 0; return; }
        
        Remove-Item $Path -Recurse *>&1 | Out-Null;
        if( $? -ne $true )
        {
            "Could not remove folder: '{0}'" -f $Path | Write-Output;
            $script:LastStatusCode = -1; return;
        }

        $script:LastStatusCode = 0; return;
    }
    function FileUtil_RemoveFiles
    {
        param( [string] $Path, [string] $Pattern )

        [bool] $PathExists = Test-Path $Path -PathType "Container";
        if( $? -ne $true )
        {
            "Could not test if folder exists: '{0}'" -f $Path | Write-Output;
            $script:LastStatusCode = -1; return;
        }
        if( !$PathExists ) { $script:LastStatusCode = 0; return; }

        # Warning: When it is used with the Include parameter, the Recurse parameter might not delete all subfolders or all child items. This is a known issue.
        # As a workaround, try piping results of the Get-ChildItem -Recurse command to Remove-Item, as described in "Example 4" in this topic.
        $Files = Get-ChildItem -Path $Path -Include $Pattern -File -Recurse;
        if( $? -ne $true )
        {
            "Could not get list of files to remove" | Write-Output;
            $script:LastStatusCode = -1; return;
        }

        [string[]] $CouldNotRemoveList = @();
        foreach( $File in $Files )
        {
            Remove-Item $File *>&1 | Out-Null;
            if( $? -ne $true ) { $CouldNotRemoveList += $File; }
        }

        if( $CouldNotRemoveList.Count -ne 0 )
        {
            "Could not remove files:`n{0}" -f $CouldNotRemoveList | Write-Output;
            $script:LastStatusCode = -1; return;
        }

        $script:LastStatusCode = 0; return;
    }
# }



class Stage
{
    [string] $Name = "default";      # stage name
    [scriptblock] $StageScript = $script:StageScript_Default;   # the stage script block to be executed
    [string[]] $CmdPartArr = @();    # the main command parts, used in some cases in the stage script
    [string[]] $CmdArgArr = @();     # +   the main command arguments
    [string] $WorkDir = $null;       # working directory that should be used to run the command
    [bool] $AcceptsArgs = $true;     # if the command accepts arguments
    [bool] $ShouldExec = $false;     # if the stage script should be executed

    # IMPORTANT: if $null is passed to string it gets converted to empty string ([string]::Empty == "")
    # +   https://github.com/PowerShell/PowerShell/issues/4616
    Stage(
        [string] $Name,
        [scriptblock] $StageScript,
        [string[]] $CmdPartArr,
        [string] $WorkDir,
        [bool] $AcceptsArgs )
    {
        $this.Name = $Name;
        $this.StageScript = $StageScript;
        $this.CmdPartArr = $CmdPartArr;
        $this.WorkDir = $WorkDir;
        $this.AcceptsArgs = $AcceptsArgs;
    }
    Stage(
        [string] $Name,
        [scriptblock] $StageScript,
        [string] $WorkDir,
        [bool] $AcceptsArgs )
    {
        $this.Name = $Name;
        $this.StageScript = $StageScript;
        $this.WorkDir = $WorkDir;
        $this.AcceptsArgs = $AcceptsArgs;
    }
    Stage(
        [string] $Name,
        [string[]] $CmdPartArr,
        [string] $WorkDir,
        [bool] $AcceptsArgs )
    {
        $this.Name = $Name;
        $this.CmdPartArr = $CmdPartArr;
        $this.WorkDir = $WorkDir;
        $this.AcceptsArgs = $AcceptsArgs;
    }

    [void] AddCmdArg( [string] $CmdArg )
    {
        $this.AddCmdArgs( $CmdArg );
    }
    [void] AddCmdArgs( [string[]] $CmdArgs )
    {
        $this.CmdArgArr += $CmdArgs;
    }

    [string] GetCommand()
    {
        return $this.GetCommandArr() -join ' ';
    }
    [string[]] GetCommandArr()
    {
        return $this.CmdPartArr + $this.CmdArgArr;
    }
}
    function Stage_ExecuteScript
    {
        param( [scriptblock] $Script, [Stage] $Stage, [bool] $PrintStageName = $true )

      # [System.Collections.ArrayList] $OutputStream = $null;
      # [System.Collections.ArrayList] $ErrorStream = $null;

      # Invoke-Command -ScriptBlock $Script -ArgumentList $Stage -OutVariable 'OutputStream' -ErrorVariable 'ErrorStream';
      # $ReturnValue = $OutputStream[ $OutputStream.Count - 1 ];
      # $OutputStream.RemoveAt( $OutputStream.Count - 1 );

      # IMPORTANT: this doesn't work as a class method because the class method's output doesn't go to the output pipeline (only the return statement's output goes to the output pipeline)
      # +   https://docs.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_classes?view=powershell-7.2#output-in-class-methods
      # $OutputStream | Write-Output;
      # $ErrorStream | Write-Error;
        
        Invoke-Command -ScriptBlock $Script -ArgumentList $Stage, $PrintStageName;
    }
# }

class Pipeline
{
    [OrderedDictionary] $StageMap = $null;

    # IMPORTANT: hashtable should be ordered! ( using the ordered attribute, e.g. [ordered]@{...} )
    Pipeline( [OrderedDictionary] $StageMap )
    {
        $this.StageMap = $StageMap;
    }

    [Stage[]] StageList()
    {
        return $this.StageMap.Values;
    }

    [Stage] Stage( [string] $StageName )
    {
        return $this.StageMap[ $StageName ];
    }

    [int] StageIdx( [string] $StageName )
    {
        $i = 0;
        foreach( $Key in $this.StageMap.Keys )
        {
            if( $Key -eq $StageName ) { return $i; }
            $i++;
        }
        return -1;
    }
}

    function Pipeline_Execute
    {
        param( [Pipeline] $Pipeline )

        foreach( $Stage in $Pipeline.StageList() )
        {
            # skip the stage if it shouldn't be executed
            if( !$Stage.ShouldExec ) { continue; }

            # set current working directory to the one specified in the pipeline stage
            # IMPORTANT: if $null is passed to string it gets converted to empty string ([string]::Empty == "")
            # +   https://github.com/PowerShell/PowerShell/issues/4616
            if( $null -ne $Stage.WorkDir   -and   "" -ne $Stage.WorkDir )
            {
                # create the working directory if it doesn't exist
                if( !( Test-Path $Stage.WorkDir -PathType "Container" ) ) { New-Item -Path $Stage.WorkDir -ItemType "Directory" | Out-Null; }

                Push-Location $Stage.WorkDir *>&1 | Out-Null;
                if( $? -ne $true )
                {
                    "Could not set the pipeline stage's working directory!" | Write-Output;
                    $script:LastStatusCode = 400; return;
                }
            }

            try
            {
                # execute the pipeline stage
                Stage_ExecuteScript $Stage.StageScript $Stage | Write-Output;
                if( $script:LastStatusCode -ne 0 ) { return; }
            }
            finally
            {
                # restore the previous working directory
                if( $null -ne $Stage.WorkDir   -and   "" -ne $Stage.WorkDir )
                {
                    Pop-Location *>&1 | Out-Null;
                    # if an error occured while restoring the previous woking directory, return
                    if( $? -ne $true )
                    {
                        "Could not restore the previous working directory!" | Write-Output;
                        $script:LastStatusCode = 400;
                    }
                }
            }

            # IMPORTANT: keep this line here since you cannot 'return' from the finally block
            if( $script:LastStatusCode -ne 0 ) { return; }
        }

        $script:LastStatusCode = 0; return;
    }
# }

# class Parser
# {
    function Parser_Parse
    {
        param( [Pipeline] $Pipeline, [string[]] $TokenArr )

        $CurrStage = $Pipeline.Stage( "=script" );
        $CurrStageIdx = 0;
        $PrevStageIdx = 0;
        $DefaultArgs_Idx = -1;

        switch -Regex ( $TokenArr )
        {
            '^='
            {
                if( $DefaultArgs_Idx -ge 0 )
                {
                    "No subcommands allowed after specifying '{0}'" -f '-def' | Write-Output;
                    $script:LastStatusCode = 400; return;
                }

                $CurrStage = $Pipeline.Stage( $_ );
                if( $null -eq $CurrStage )
                {
                    "Unknown subcommand: '{0}'" -f $_ | Write-Output;
                    $script:LastStatusCode = 400; return;
                }

                $CurrStageIdx = $Pipeline.StageIdx( $_ );
                if( $CurrStageIdx -le $PrevStageIdx )
                {
                    "Invalid placement for subcommand '{0}'; view help for subcommand ordering." -f $_ | Write-Output;
                    $script:LastStatusCode = 400; return;
                }

                $CurrStage.ShouldExec = $true;
                continue;
            }
            default
            {
                if( $CurrStageIdx -eq 0 )
                {
                    if( $_ -match '-\d+' )
                    {
                        if( $DefaultArgs_Idx -ge 0 )
                        {
                            "Cannot specify more than one default parameter list: '{0}'" -f $_ | Write-Output;
                            $script:LastStatusCode = 400; return;
                        }

                        $DefaultArgs_Idx = -( $_ -as [int] );
                        if( $DefaultArgs_Idx -lt 0   -or  $DefaultArgs_Idx -ge $script:DefaultArgs.Count )
                        {
                            "Unknown parameter: '{0}'" -f $_ | Write-Output;
                            $script:LastStatusCode = 400; return;
                        }

                        continue;
                    }

                    $CurrStage.ShouldExec = $true;
                }

                $CurrStage.AddCmdArg( $_ );
                continue;
            }
        }

        if( $DefaultArgs_Idx -ge 0 )
        {
            Parser_Parse $Pipeline $script:DefaultArgs[ $DefaultArgs_Idx ] | Write-Output;
            return;
        }
        
        $script:LastStatusCode = 0; return;
    }
# }





# ________________________________________________________________________________________________________________
# Resources

[Stage] $script:ScriptStg = [Stage]::new(
    "SCRIPT PARAMS",
    {
        param( [Stage] $Stage )
    
        # switch script parameters
        switch( $Stage.GetCommandArr() )
        {
            "--help"
            {
                $script:HelpMessage | Write-Output;
                continue;
            }
            "-help"
            {
                $script:HelpMessage | Write-Output;
                continue;
            }
            default
            {
                "Unknown parameter: {0}" -f $_ | Write-Output;
                $script:LastStatusCode = -1; return;
            }
        }

        $script:LastStatusCode = 0; return;
    },
    $null,
    $true
);

[Stage] $script:ProjCleanStg = [Stage]::new(
    "PROJECT CLEAN",
    {
        param( [Stage] $Stage )

        # print the stage name
        $script:StageSep -f $Stage.Name | Write-Output;

        # get the command arguments from the stage
        $StageCommandArr = $Stage.CmdArgArr;

        # clean command parameters
        [bool] $ShouldCleanMisc = $false;


        # switch the clean command parameters
        switch( $StageCommandArr )
        {
            "-all"
            {
                $ShouldCleanMisc = $true;
                continue;
            }
            default
            {
                "Unknown parameter: {0}" -f $_ | Write-Output;
                $script:LastStatusCode = -1; return;
            }
        }


        # clean the build items (and logs if necessary)
        $BuildItems = @(
            # remove the build directory
            [PSCustomObject]@{ Path="./build"; Filter=""; }
            # remove the .vs directory
            [PSCustomObject]@{ Path ="./.vs"; Filter=""; }
            # remove the VS's temp folder
            [PSCustomObject]@{ Path ="./enc_temp_folder"; Filter=""; }
        );
        $MiscItems = @(
            # remove the logs in the log directory
            [PSCustomObject]@{ Path ="./log"; Filter=""; }
        );

        $ItemsToRemove = @();
        if( $true            ) { $ItemsToRemove = $ItemsToRemove + $BuildItems; }
        if( $ShouldCleanMisc ) { $ItemsToRemove = $ItemsToRemove + $MiscItems;  }

        foreach( $Item in $ItemsToRemove )
        {
            '[clean] Path: "{0}" Filter: "{1}"' -f $Item.Path, $Item.Filter | Write-Output;
            
            if( "" -eq $Item.Filter ) { FileUtil_RemoveFolder $Item.Path | Write-Output; }
            else                      { FileUtil_RemoveFiles $Item.Path $Item.Filter | Write-Output; }

            if( $script:LastStatusCode -ne 0 ) { return; }
        }

        $script:LastStatusCode = 0; return;
    },
    "${script:ProjectRoot}/",
    $false
);

# build project
[Stage] $script:ProjBuildStg = [Stage]::new(
    "PROJECT BUILD",
    {
        param( [Stage] $Stage )

        # print the stage name
        $script:StageSep -f $Stage.Name | Write-Output;

        # get the command arguments from the stage
        $StageCommandArr = $Stage.CmdArgArr;

        # build command parameters
        [string] $Optimization = '-O2';


        # switch the build command parameters
        switch( $StageCommandArr )
        {
            "-debug"
            {
                $Optimization = '-DDEBUG';
                continue;
            }
            default
            {
                "Unknown parameter: {0}" -f $_ | Write-Output;
                $script:LastStatusCode = -1; return;
            }
        }

        # all .cpp and .cu source files
        $SourceFiles  = @();
        $SourceFiles += @(FileUtil_FindRecursive -Location "." -Extension '.cpp');
        $SourceFiles += @(FileUtil_FindRecursive -Location "." -Extension '.cu' );
        # if there aren't any source files, exit
        if( $SourceFiles.Count -eq 0   -and   $script:LastStatusCode -eq 0 )
        {
            "No source files given" | Write-Output;
            $script:LastStatusCode = -1; return;
        }

        # determine all other parameteres
        if( $isWindows )
        {
            # set the openmp flag for the compiler (visual studio c++ on windows or gcc on linux)
            $Openmp = '/openmp';
        }
        else
        {
            # set the openmp flag for the compiler (visual studio c++ on windows or gcc on linux)
            $Openmp = '-openmp';
        }


        # create the build command
        $Stage.CmdPartArr = 'nvcc',      # call the nvidia c++ compiler wrapper
            '-Xcompiler', '"',           # pass the string arguments to the underlying c++ compiler (msvc)
                "$Openmp",               # +   use openmp
            '"',                         # 
            "--std=c++17",               # set the c++ standard (currently nvcc doesn't support c++20)
            '--ptxas-options=-v',        # show verbose cuda kernel compilation info
            '-arch=sm_61',               # architecture - cuda 6.1
          # "-use_fast_math",            # use fast math
            '-maxrregcount 32',          # maximum registers available per thread
            "$Optimization",             # define the debug symbol, or optimise code, depending on what is requested
            "   -o `"../build/pwsh/${script:Target}`"", # add the output file name to the build command
            $SourceFiles;                # source files

        # create the output directory if it doesn't exist
        if( !( Test-Path "../build/pwsh" -PathType "Container" ) ) { New-Item -Path "../build/pwsh" -ItemType "Directory" | Out-Null; }

        # invoke the default stage script on this stage
        Stage_ExecuteScript $script:StageScript_Default $Stage $false | Write-Output;
    },
    "${script:ProjectRoot}/src",
    $false
);
# run the exe
[Stage] $script:NwRunStg = [Stage]::new(
    "NW RUN",
    @(
        # NOTE: command paths should be relative to the project 'build/pwsh' folder
        "./${script:Target}"
    ),
    "${script:ProjectRoot}/build/pwsh",
    $true
);

[Pipeline] $script:Pipeline = [Pipeline]::new(
    [ordered]@{
        "=script"  = $ScriptStg
        "=clean"   = $ProjCleanStg
        "=build"   = $ProjBuildStg
        "=run"     = $NwRunStg
    }
);





# ________________________________________________________________________________________________________________
# Script

# main function
function Build-V2
{
    $ScriptArgs = $args.Count -ne 0 ? $args : $( "--help" );

    Parser_Parse $script:Pipeline $ScriptArgs;
    if( $script:LastStatusCode -ne 0 ) { return; }
    
    Pipeline_Execute $script:Pipeline;
    if( $script:LastStatusCode -ne 0 ) { return; }
}

# call the build script
# +   @ - array splatting operator; used here to pass script arguments to the build function
Build-V2 @args *>&1 | Tee-Object -FilePath "${script:ProjectRoot}/build.log" -Append;
# exit with the last exit code
exit $script:LastStatusCode;





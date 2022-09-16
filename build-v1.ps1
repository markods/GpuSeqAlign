# ________________________________________________________________________________________________________________
# Reference materials

# https://docs.microsoft.com/en-us/powershell/scripting/learn/deep-dives/everything-about-hashtable?view=powershell-7.1
# https://docs.microsoft.com/en-us/powershell/scripting/learn/deep-dives/everything-about-arrays?view=powershell-7.1
# https://docs.microsoft.com/en-us/powershell/scripting/learn/deep-dives/everything-about-hashtable?view=powershell-7.1#saving-a-nested-hashtable-to-file
# https://docs.microsoft.com/en-us/powershell/scripting/learn/ps101/06-flow-control?view=powershell-7.1
# https://docs.microsoft.com/en-us/powershell/scripting/learn/ps101/09-functions?view=powershell-7.1

# https://docs.microsoft.com/en-us/dotnet/api/system.collections.arraylist?view=net-5.0
# https://powershellexplained.com/2017-11-20-Powershell-StringBuilder/
# https://docs.microsoft.com/en-us/powershell/scripting/learn/deep-dives/everything-about-null?view=powershell-7.1
# https://docs.microsoft.com/en-us/powershell/module/microsoft.powershell.management/remove-item?view=powershell-7.1#example-4--delete-files-in-subfolders-recursively
# https://docs.microsoft.com/en-us/powershell/scripting/learn/deep-dives/everything-about-switch?view=powershell-7.1
# https://docs.microsoft.com/en-us/powershell/scripting/learn/deep-dives/everything-about-hashtable?view=powershell-7.1#splatting-hashtables-at-cmdlets
# https://docs.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_operators?view=powershell-7.1



# ________________________________________________________________________________________________________________
# Functions

# find all files with the given extension recursively relative to the given location
function Find-Recursive
{
    # this turns a regular function into an advanced function
    [CmdletBinding()]
    # function parameters
    param( [string] $Extension )

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
            Push-Location "./build/pwsh";
            $Result = @( $Files.FullName | Resolve-Path -Relative | ForEach-Object -Process { $_.ToString() } );
        }
        finally
        {
            Pop-Location;
        }
    }

    return $Result;
}



# ________________________________________________________________________________________________________________
# Build script

if( "-help" -in $args -or "--help" -in $args )
{
    Write-Output "build   [[-]-help]   [=clean] [=build]   [=run scores params seqs\n]";
    Write-Output "";
    Write-Output "Default:        build =build";
    Write-Output "";
    Write-Output "Switches:";
    Write-Output "    --help      shows the help menu";
    Write-Output "    -help       same as --help";
    Write-Output "";
    Write-Output "    =clean      cleans the project";
    Write-Output "    =build      builds the project";
    Write-Output "      -debug    +   don't optimise and define the DEBUG symbol";
    Write-Output "";
    Write-Output "    =run        runs the compiled program";
    Write-Output "      scores    +   path to json file (relative to /resrc) with score matrices\n";
    Write-Output "      params    +   path to json file (relative to /resrc) with parameters for the Needleman-Wunsch algorithms\n";
    Write-Output "      seqs      +   path to json file (relative to /resrc) with sequences to be compared\n";
    Write-Output "";
    exit 0;
}


# default build script arguments if none specified
if( $args.count -eq 0 )
{
    # leave powershell array constructor ( @() ) even if there is only one argument (otherwise it won't be a powershell array due to unpacking)
    $args = @( "=build" );
}

# calculate the "run" command's position in the argument list
$BuildArgs = $null;
$RunArgs = $null;

if( $true )
{
    $BuildArgs_Beg, $BuildArgs_End = [array]::indexof( $args, "=build" ), $args.count;
    $RunArgs_Beg,   $RunArgs_End   = [array]::indexof( $args, "=run" ),   $args.count;

    if( $BuildArgs_Beg -ge 0 )
    {
        if( $RunArgs_Beg    -ge 0 ) { $BuildArgs_End = $RunArgs_Beg; }

        if( $BuildArgs_Beg + 1   -le   $BuildArgs_End - 1 ) { $CompileArgs = $args[ ($BuildArgs_Beg + 1) .. ($BuildArgs_End - 1) ]; }
    }
    if( $RunArgs_Beg -ge 0 )
    {
        if( $RunArgs_Beg + 1   -le   $RunArgs_End - 1 ) { $RunArgs = $args[ ($RunArgs_Beg + 1) .. ($RunArgs_End - 1) ]; }
    }
}

# compilation target
$Target = 'nw.exe';



# clean project
if( "=clean" -in $args )
{
    Write-Output "---------------------------------------------------------------------------------------------------------------- <<< CLEAN";

    # remove the compiled code directory
    if( Test-Path "./build" -PathType "Container" ) { Remove-Item "./build" -Recurse; }

    # print the build command
    Write-Output "Clean success\n"
}



# build project
if( "=build" -in $args )
{
    Write-Output "---------------------------------------------------------------------------------------------------------------- <<< BUILD";

    # create the build folder if it doesn't exist
    if( !( Test-Path "./build/pwsh" -PathType "Container" ) ) { New-Item -Path "./build/pwsh" -ItemType "Directory" | Out-Null; }

    # all .cpp and .cu source files
    $SourceFiles = $null;
    $SourceFiles = @(Find-Recursive -Extension '.cu') + @(Find-Recursive -Extension '.cpp');
    # if there aren't any source files, exit
    if( $SourceFiles.Count -eq 0 ) { "No source files given"; exit -1; }


    if( $isWindows )
    {
        # set the c++ compiler standard
        $StdCpp = '/std:c++20';
        # set the openmp flag for the compiler (visual studio c++ on windows or gcc on linux)
        $Openmp = '/openmp';
    }
    else
    {
        # set the c++ compiler standard
        $StdCpp = '-std:c++20';
        # set the openmp flag for the compiler (visual studio c++ on windows or gcc on linux)
        $Openmp = '-openmp';
    }
    # if debugging is requested, don't optimize the code
    $Debug = ( '-debug' -in $args ) ? '-DDEBUG' : '-O2';

    # create the build command
    $BuildCmd = $null;
    $BuildCmd = 'nvcc',            # call the nvidia c++ compiler wrapper
        "$StdCpp",                 # use the c++20 standard
        '-Xcompiler', '"',         # pass the string arguments to the underlying c++ compiler (msvc)
            "$Openmp",             # +   use openmp
        '"',                       # 
        '--ptxas-options=-v',      # show verbose cuda kernel compilation info
        '-arch=sm_61',             # architecture - cuda 6.1
      # '-prec-sqrt true',         # use precise sqrt
        '-maxrregcount 32',        # maximum registers available per thread
        "$Debug",                  # define the debug symbol, or optimise code, depending on what is requested
        '   -o "{0}" ' -f $Target; # add the output file name to the build command

    # join the array of strings into a single string using the space character
    $BuildCmd = $BuildCmd -Join ' ';
    # append the source files to the command
    $BuildCmd += '"' + ($SourceFiles -Join '" "') + '"';

    # print the build command
    Write-Output $BuildCmd;

    try
    {
        # set current working directory to the project 'build' folder
        Push-Location "./build/pwsh";

        # invoke the build command
        Invoke-Expression -Command $BuildCmd;
    }
    finally
    {
        # restore the previous working directory
        Pop-Location;
    }

    # exit with the error code of the last native program that was run
    if( $LASTEXITCODE -ne 0 ) { exit $LASTEXITCODE; }
}



# run the compiled code
if( "=run" -in $args )
{
    Write-Output "---------------------------------------------------------------------------------------------------------------- <<< RUN";

    # always initialize variables before use (since an exception can occur during assignment)
    $RunCmd = $null;
    # create the run comand array
    $RunCmd = ,"./$Target";

    if( $RunArgs.count -gt 0 ) { $RunCmd += $RunArgs; }

    # join the array of strings into a single string separated by spaces
    $RunCmd = $RunCmd -Join ' ';

    # print the run command
    Write-Output $RunCmd;

    try
    {
        # set current working directory to the project 'build' folder to simplify paths
        Push-Location "./build/pwsh";
    
        # invoke the run command
        Invoke-Expression -Command $RunCmd;
    }
    finally
    {
        # restore the previous working directory
        Pop-Location;
    }

    # exit with the error code of the last native program that was run
    if( $LASTEXITCODE -ne 0 ) { exit $LASTEXITCODE; }
}



# exit success
exit 0;






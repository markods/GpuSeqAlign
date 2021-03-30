# find all files with the given extension recursively relative to the current working directory
function Find-Recursive
{
    # this turns a regular function into an advanced function
    [CmdletBinding()]
    # function parameters
    param( [string] $Extension )
    
    # always initialize variables before use (since an exception can occur during assingment)
    $Files = $null;
    # find files with the given extension recursively in the current working directory
    $Files = Get-ChildItem -Path "$PWD" -Include "*$Extension" -File -Recurse;

    # for all files, get the full name from the file info, get the path relative to the current working directory,
    # and finally convert the result to a string
    # +   return an array of such files
    if( $Files ) { @(   $Files.FullName | Resolve-Path -Relative | ForEach-Object -Process { $_.ToString() }   ); }
    else         { @(); }
}


# if compilation is requested (if generated files should not be removed)
if( '-Clear' -notin $args ) {
    # compilation target
    $Target = 'needle.exe';
    
    # all .cpp and .cu source files
    $SourceFiles = $null;
    $SourceFiles = @(Find-Recursive -Extension '.cu') + @(Find-Recursive -Extension '.cpp');
    # if there aren't any source files, exit
    if( $SourceFiles.Count -eq 0 ) { "No sources available"; exit 0; }

    # build the compiler command
    $CompilerCmd = $null;
    $CompilerCmd = 'nvcc',       # call the nvidia c++ compiler wrapper
        '-Xcompiler', '"',       # pass the string arguments to the underlying c++ compiler (msvc)
            '/openmp',           # +   use openmp
          # '/GL',               # +   use whole program optimization
                                 #     +   TODO: this option messes something up in the cuda kernel!?
        '"',                     # 
        '-Xptxas -v',            # show verbose cuda kernel compilation info
        '-arch=sm_61',           # architecture - cuda 6.1
        '-prec-sqrt true',       # use precise sqrt
        '-maxrregcount 32';      # maximum registers available per thread
    
    # if debugging is requested, don't optimize the code
    if( '-Debug' -in $args ) { $CompilerCmd += '-DDEBUG'; }
    else                     { $CompilerCmd += '-O2'; }

    # add the output file nameto the compiler command
    $CompilerCmd += '   -o "{0}" ' -f $Target;
    # join the array of strings into a single string using the space character
    $CompilerCmd = $CompilerCmd -Join ' ';
    # append the source files to the command
    $CompilerCmd += '"' + ($SourceFiles -Join '" "') + '"';

    # print the compiler command
    $CompilerCmd
    
    # invoke the compiler command
    Invoke-Expression -Command $CompilerCmd
}


# get temporary generated files (with the .lib and .exp extensions)
$TempFiles = $null;
$TempFiles = @(Find-Recursive -Extension '.lib') + @(Find-Recursive -Extension '.exp');

# if all project generated files should be removed
if( '-Clear' -in $args ) {
    # add some more generated files to the list
    $TempFiles += @(Find-Recursive -Extension '.exe') + @(Find-Recursive -Extension '.out*.txt');
    # clear the terminal
    Clear-Host;
}

# remove all such temporary files
if( $TempFiles.Count -gt 0 ) { $TempFiles | Remove-Item; }

# exit success
exit 0;






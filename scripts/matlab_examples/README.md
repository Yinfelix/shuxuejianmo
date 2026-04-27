# MATLAB Or Octave In VS Code

## Can This Workspace Use MATLAB Syntax?

Yes.

- `.m` files can be edited directly in VS Code.
- The installed MATLAB language extension gives syntax highlighting, basic navigation, and editing support.
- If you install local MATLAB, you can run scripts from the VS Code terminal.
- If you install GNU Octave, most contest-style matrix and algorithm code can also run with minor adjustments.

## Current Machine State

- MATLAB command is not currently available on PATH.
- Octave command is not currently available on PATH.

That means this workspace can already write MATLAB syntax, but it cannot execute `.m` files until one of those runtimes is installed.

This workspace now also includes a PowerShell launcher at `scripts/run_octave.ps1`.
Octave is unpacked locally under `tools/octave/octave-11.1.0-w64`.
The VS Code task `Run current .m file (Octave)` can execute the current `.m` file directly.
The integrated terminal for this workspace also receives the local Octave `mingw64/bin` directory on `PATH`.

## Example Workflow

1. Open `scripts/matlab_examples/topsis_demo.m`.
1. Modify the matrix, weights, and criterion directions for your contest problem.
1. If MATLAB is installed, run from terminal:

```powershell
matlab -batch "run('scripts/matlab_examples/topsis_demo.m')"
```

1. If Octave is installed, run from terminal:

```powershell
octave --quiet --eval "addpath('scripts/matlab_examples'); topsis_demo"
```

## When To Prefer MATLAB Or Octave

- Use MATLAB or Octave when the problem is matrix-heavy and you want very fast numerical prototyping.
- Keep Python as the safer default when you need stronger package ecosystems, notebooks, machine learning, or easier environment sharing.
- For school contests, evaluation models, regression, interpolation, optimization prototypes, Monte Carlo simulation, and graph algorithms can all be written in MATLAB-style syntax.

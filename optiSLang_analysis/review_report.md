# Motor-CAD optiSLang Folder Review

Target folder:
- C:/Program Files/ANSYS Inc/v261/motorcad/Motor-CAD Data/optiSLang

## Reviewed Files and Roles

1. optimisationScript_Header.py
- Defines runtime constants and utility functions.
- Implements Logger and output file writing helpers.
- Implements `RunOptimisation(aRunMode)` startup, Motor-CAD connection, parameter assignment, geometry/winding checks, and calculation kickoff.
- Provides validation/error paths that return NaN outputs when design is invalid.

2. optimisationScript_Footer.py
- Completes solver flow after calculations.
- Saves final Motor-CAD design, releases/quits instance, and writes `MotorCAD_Outputs.txt`.
- Contains runtime entry block that decides run mode (Python node / IDE / internal test), initializes inputs/outputs, and invokes `RunOptimisation`.

3. optislangSetupScript.py
- Builds an optiSLang project programmatically.
- Creates Motor-CAD integration node, applies Python executable settings, loads discovered parameters/responses, and launches wizard.
- Reads `MotorCAD_Summary.txt` to configure optimization parameters and response criteria/objectives/constraints.

4. Integration/MotorCAD_ci.py
- Custom integration plugin API layer for optiSLang.
- Exposes required callbacks:
  - `DefaultSettings`
  - `ExtractInputContainerForDesigns`
  - `SetParametersForDesigns`
  - `RunSolverForDesigns`
  - `ExtractOutputContainerForDesigns`
- Delegates implementation details to `MotorCAD_integration.py` and `MotorCAD_settings.py`.

5. Integration/MotorCAD/MotorCAD_integration.py
- Core bridge logic.
- `scan_input`: scans input section markers and reads `i_*` parameters.
- `set_parameters`: writes per-design script with updated parameter values and adjusted reference directory.
- `scan_output`: finds output results file declaration and parses `MotorCAD_Outputs.txt`.
- `run_Python_Script`: prepares environment and executes `RunOptimisation(0)` as subprocess.

6. Integration/MotorCAD/MotorCAD_settings.py
- Defines integration settings objects and defaults for Input/Execution/Output tabs.
- Supports selecting a custom Python executable (`Execution.use_other_python`, `Execution.otherpy_exe`).

7. Integration/MotorCAD/MotorCAD_helpers.py
- Utility helpers for duplicate-name detection, environment variable export, subprocess execution, and debug printouts.

8. Integration/MotorCAD_ci.cfg
- Integration runtime behavior configuration:
  - `EnableParallel true`
  - `ScriptInterfaceVersion 2`
  - `EnableMultiDesignMode false`
  - `SupportsHPCLicenseContext true`

## End-to-End Runtime Flow

1. Export/setup phase
- `optislangSetupScript.py` creates `.opf` project and Motor-CAD integration node.
- It parses summary definitions and registers parameters/responses in optiSLang.

2. Per-design execution phase
- optiSLang calls `MotorCAD_ci.py` callbacks.
- Input extraction: `scan_input` identifies editable `i_*` variables in solver script.
- Parameter set: `set_parameters` writes design-specific solver script.
- Solver run: `run_Python_Script` launches Python subprocess and calls `RunOptimisation(0)`.

3. Solver internal phase
- `RunOptimisation` in header/footer script configures Motor-CAD, applies parameters, runs checks, executes calculations, and writes outputs file.

4. Output extraction phase
- `scan_output` parses `MotorCAD_Outputs.txt` and returns values to optiSLang.

## Generated Diagram

- UML-style activity PNG:
  - `optiSLang_analysis/MotorCAD_optiSLang_UML_flow.png`

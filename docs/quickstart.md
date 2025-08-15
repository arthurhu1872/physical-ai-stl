# Quickstart

## Environment setup

Create and activate a virtual environment (recommended) to keep dependencies isolated:

    python3 -m venv .venv
    source .venv/bin/activate   # on Windows use .\\.venv\\Scripts\\Activate.ps1

Install the package and dependencies in editable mode:

    pip install --upgrade pip
    pip install -e .

This installs the required packages listed in 'pyproject.toml'. Optional frameworks (Neuromancer, PhysicsNeMo, TorchPhysics, SpaTiaL, etc.) can be checked via:

    make env   # quick dependency check
    make survey  # show versions of installed physical-AI frameworks

### Java requirement

MoonLight requires a Java runtime. Install Java 21 or higher and ensure 'java -version' reports 21.x. On macOS you can use Homebrew:

    brew install openjdk@21
    java -version

On Windows you can use the Microsoft OpenJDK via winget:

    winget install --id Microsoft.OpenJDK.21
    java -version

Ensure the 'JAVA_HOME' environment variable points to the JDK 21 installation and that 'java' is on your 'PATH'.

## Running examples

You can run the example monitors from the 'src/physical_ai_stl/monitors' package as standalone scripts:

    python -m physical_ai_stl.monitors.rtamt_hello
    python -m physical_ai_stl.monitors.moonlight_hello

These commands compute robustness for simple temporal and spatio-temporal specifications and print the results.

## Running tests

The repository uses 'pytest' for tests. To run all tests:

    pytest -q

This will execute the smoke tests in the 'tests' directory.

---

For more information on Signal Temporal Logic (STL) and spatio-temporal specifications, see the documentation for [RTAMT](https://github.com/nickovic/rtamt) and [MoonLight](https://pypi.org/project/moonlight/).

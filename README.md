# physical-ai-stl

Scaffold for experimenting with **Signal Temporal Logic (STL)** and **spatio‑temporal logic (STREL)** monitoring in the context of physics‑based machine learning models It's intended to be a **teaching and prototyping repository**, not a production‑ready library The goal is to make it easy to tinker with temporal/spatio‑temporal specifications alongside simple neural ODE or PDE examples It forms part of a larger initiative to explore "physical AI" — blending physics‑informed models (neural ODEs/PDEs, PINNs) with formal specification monitoring to ensure safety, stability and robustness in AI‑driven systems.

## Status

This repository is under active exploration and is **not** an officially supported package The examples and scripts here are primarily for educational and research use The APIs and file structure may change as the project evolves If you intend to build on top of this code, please pin dependency versions and expect that future commits might break backwards compatibility.

## Overview

This codebase bootstraps an environment to prototype and test logic‑based monitors before integrating them into more complex frameworks such as [Neuromancer](https://github.com/pnnl/neuromancer), [TorchPhysics](https://github.com/boschresearch/torchphysics) or [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) It includes:

* A modern Python project layout (`pyproject.toml`) with dependencies like [RTAMT](https://github.com/nickovic/rtamt) for STL monitoring and [MoonLight](https://github.com/MoonLightSuite/moonlight) for spatio‑temporal logic.
* Example modules under `src/physical_ai_stl/monitors` demonstrating a simple offline STL robustness check and a spatio‑temporal monitor These illustrate how to call the monitors programmatically and can serve as a template for your own experiments.
* A test suite (`tests/`) for the example monitors, executed automatically via PyTest.
* A GitHub Actions workflow (`.github/workflows/ci.yml`) that installs Java 21, sets up Python, installs dependencies and runs the tests on each push or pull request.

## Quickstart

Follow these steps to set up your environment and run the examples:

1. **Clone the repo** and create a virtual environment (Python 3.9+) A dedicated environment keeps your global Python installation clean and avoids version conflicts:

   ```bash
   git clone https://github.com/arthurhu1872/physical-ai-stl.git
   cd physical-ai-stl
   python3 -m venv .venv
   # On Windows use `.\\.venv\\Scripts\\activate`
   source .venv/bin/activate
   ```

2. **Install dependencies** and the package in editable mode Installing with `-e` (editable) allows you to modify the code in place and immediately see the effects when you run the scripts:

   ```bash
   pip install --upgrade pip
   pip install -e .
   ```

   This will install `rtamt`, `moonlight`, `pytest` and other dependencies specified in `pyproject.toml`.

3. **Verify you have Java 21+ available for MoonLight** The MoonLight Python wrapper relies on a recent Java runtime; versions prior to JDK 21 are not supported:

   ```bash
   java -version
   ```

   If the version is lower than 21, follow your operating system's instructions to install [OpenJDK 21](https://jdk.java.net/21/) or a compatible distribution.

4. **Run the example scripts** These scripts demonstrate a basic offline STL evaluation (via RTAMT) and a spatio‑temporal evaluation (via MoonLight):

   ```bash
   # Evaluate a simple STL specification using RTAMT
   python -m physical_ai_stl.monitors.rtamt_hello

   # Evaluate a simple temporal and spatio‑temporal specification using MoonLight
   python -m physical_ai_stl.monitors.moonlight_hello
   ```

5. **Run the test suite** (optional but recommended):

   ```bash
   pytest -q
   ```

6. **Continuous Integration** The repository includes a GitHub Actions workflow that automatically sets up the environment and runs tests whenever you push or open a pull request This helps ensure that the example monitors continue to work as dependencies evolve and makes it easier to verify contributions.

## Contributing

Contributions are welcome! Please open an issue or discussion first to talk about significant changes you'd like to make. For small fixes, feel free to submit a pull request directly. Make sure to run the tests and update the documentation where appropriate.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

The initial version of this repository was created as part of an undergraduate research project exploring logic‑guided training for neural ODE/PDE models. It builds on the excellent work of the RTAMT and MoonLight authors and the broader physics‑informed machine learning community.

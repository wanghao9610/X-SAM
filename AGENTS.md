# Agent Instructions
> Use this Python environment for all Python execution, debugging, tests, and dependency checks.

**Before running any Python or Conda command**, read the project root `.env` file and resolve:

If `.env` is missing, run `cp .env.example .env` and set machine-specific values. The repo tracks `.env.example` only; `.env` is gitignored.

- `CONDA_HOME` — Conda installation root (all Conda paths derive from this)
- `PYTHON_HOME` — active Conda env root (Python interpreter and env-specific paths derive from this)

Do not hardcode machine-specific paths. Do not use system Python.

Derived paths (replace placeholders with values from `.env`):

| Purpose | Path |
|--------|------|
| Conda executable | `{CONDA_HOME}/bin/conda` |
| Conda root | `{CONDA_HOME}` |
| Conda environment name | basename of `{PYTHON_HOME}` (e.g. `xchat`) |
| Python interpreter | `{PYTHON_HOME}/bin/python` |

When running Python commands, prefer:

`{CONDA_HOME}/bin/conda run -n <env_name> python ...`

where `<env_name>` is the basename of `PYTHON_HOME`.

For tests, prefer:

`{CONDA_HOME}/bin/conda run -n <env_name> pytest`

For direct interpreter execution, use:

`{PYTHON_HOME}/bin/python ...`

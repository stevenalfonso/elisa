# Running Tests

## Setup

Install the package with dev dependencies:

```bash
pip install -e ".[dev]"
```

If `ezpadova` is not available for your Python version, install without dependencies and add the core ones manually:

```bash
pip install -e . --no-deps
pip install numpy pandas scipy emcee matplotlib pytest
```

## Run all tests

```bash
pytest
```

With verbose output:

```bash
pytest -v
```

## Run a specific test file

```bash
pytest tests/test_utils.py
pytest tests/test_extinction.py
pytest tests/test_interpolator.py
pytest tests/test_posterior.py
pytest tests/test_elisa.py
```

## Run a specific test class or method

```bash
pytest tests/test_elisa.py::TestGelmanRubin
pytest tests/test_elisa.py::TestGelmanRubin::test_converged_chains
```

## Test structure

```
tests/
├── conftest.py            # Shared fixtures (synthetic grid, observed data)
├── test_utils.py          # Priors, IMF, log-likelihood
├── test_extinction.py     # Distance conversions, extinction law
├── test_interpolator.py   # Isochrone interpolation
├── test_posterior.py      # Prior evaluation, posterior, mass estimation
└── test_elisa.py          # ElisaClusterInference end-to-end
```

All tests use a small synthetic isochrone grid defined in `conftest.py` and do not require network access or downloaded data files.

## Adding new tests

1. Create a function prefixed with `test_` inside an existing file, or create a new `tests/test_<module>.py` file.
2. Use the `synthetic_grid` and `observed_data` fixtures from `conftest.py` when you need isochrone or photometry data.
3. Run `pytest -v` to verify.

import pytest
import pandas as pd
import wandb

def pytest_addoption(parser):
    """Add CLI options for pytest."""
    parser.addoption("--input_artifact", action="store", help="Input CSV file to be tested")
    parser.addoption("--output_artifact", action="store", help="Reference CSV file to compare the new CSV to")
    parser.addoption("--kl_threshold", action="store", help="Threshold for KL divergence")
    parser.addoption("--min_rows", action="store", help="Minimum number of rows for validation")
    parser.addoption("--max_rows", action="store", help="Maximum number of rows for validation")
    parser.addoption("--min_price", action="store", help="Minimum price for validation")
    parser.addoption("--max_price", action="store", help="Maximum price for validation")

@pytest.fixture(scope='session')
def wandb_run():
    """Initialize a single W&B run for session-wide use."""
    run = wandb.init(job_type="data_tests", settings=wandb.Settings(start_method="thread"))
    yield run  # Keeps it open for all fixtures
    run.finish()  # Ensure cleanup after session ends

@pytest.fixture(scope='session')
def data(request, wandb_run):
    """Download input artifact from W&B."""
    artifact_name = request.config.getoption("--input_artifact")
    if not artifact_name:
        pytest.fail("Error: --input_artifact is missing.")

    try:
        data_path = wandb_run.use_artifact(artifact_name).file()
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        pytest.fail(f"Failed to fetch input artifact '{artifact_name}': {e}")

@pytest.fixture(scope='session')
def ref_data(request, wandb_run):
    """Download reference (output) artifact from W&B."""
    artifact_name = request.config.getoption("--output_artifact")
    if not artifact_name:
        pytest.fail("Error: --output_artifact is missing.")

    try:
        data_path = wandb_run.use_artifact(artifact_name).file()
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        pytest.fail(f"Failed to fetch output artifact '{artifact_name}': {e}")

@pytest.fixture(scope='session')
def kl_threshold(request):
    value = request.config.getoption("--kl_threshold")
    if value is None:
        pytest.fail("Error: --kl_threshold is missing.")
    return float(value)

@pytest.fixture(scope='session')
def min_rows(request):
    value = request.config.getoption("--min_rows")
    if value is None:
        pytest.fail("Error: --min_rows is missing.")
    return int(value)

@pytest.fixture(scope='session')
def max_rows(request):
    value = request.config.getoption("--max_rows")
    if value is None:
        pytest.fail("Error: --max_rows is missing.")
    return int(value)

@pytest.fixture(scope='session')
def min_price(request):
    value = request.config.getoption("--min_price")
    if value is None:
        pytest.fail("Error: --min_price is missing.")
    return float(value)

@pytest.fixture(scope='session')
def max_price(request):
    value = request.config.getoption("--max_price")
    if value is None:
        pytest.fail("Error: --max_price is missing.")
    return float(value)
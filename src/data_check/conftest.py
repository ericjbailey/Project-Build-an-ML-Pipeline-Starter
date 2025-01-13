import pytest
import pandas as pd
import wandb


def pytest_addoption(parser):
    parser.addoption("--input_artifact", action="store", help="Input CSV file to be tested")
    parser.addoption("--output_artifact", action="store", help="Reference CSV file to compare the new CSV to")
    parser.addoption("--kl_threshold", action="store", help="Threshold for KL divergence")
    parser.addoption("--min_rows", action="store", help="Minimum number of rows for validation")
    parser.addoption("--max_rows", action="store", help="Maximum number of rows for validation")
    parser.addoption("--min_price", action="store", help="Minimum price for validation")
    parser.addoption("--max_price", action="store", help="Maximum price for validation")


@pytest.fixture(scope='session')
def data(request):
    run = wandb.init(job_type="data_tests", resume=True)

    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    data_path = run.use_artifact(request.config.getoption("--input_artifact")).file()

    if data_path is None:
        pytest.fail("You must provide the --input_artifact option on the command line")

    df = pd.read_csv(data_path)

    return df


@pytest.fixture(scope='session')
def ref_data(request):
    run = wandb.init(job_type="data_tests", resume=True)

    # Download output artifact. This will also note that this script is using this
    # particular version of the artifact
    data_path = run.use_artifact(request.config.getoption("--output_artifact")).file()

    if data_path is None:
        pytest.fail("You must provide the --output_artifact option on the command line")

    df = pd.read_csv(data_path)

    return df


@pytest.fixture(scope='session')
def kl_threshold(request):
    kl_threshold = request.config.getoption("--kl_threshold")

    if kl_threshold is None:
        pytest.fail("You must provide a threshold for the KL test")

    return float(kl_threshold)


@pytest.fixture(scope='session')
def min_rows(request):
    min_rows = request.config.getoption("--min_rows")

    if min_rows is None:
        pytest.fail("You must provide min_rows")

    return int(min_rows)


@pytest.fixture(scope='session')
def max_rows(request):
    max_rows = request.config.getoption("--max_rows")

    if max_rows is None:
        pytest.fail("You must provide max_rows")

    return int(max_rows)


@pytest.fixture(scope='session')
def min_price(request):
    min_price = request.config.getoption("--min_price")

    if min_price is None:
        pytest.fail("You must provide min_price")

    return float(min_price)


@pytest.fixture(scope='session')
def max_price(request):
    max_price = request.config.getoption("--max_price")

    if max_price is None:
        pytest.fail("You must provide max_price")

    return float(max_price)
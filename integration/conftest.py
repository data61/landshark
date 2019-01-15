"""Configuration for test suite."""

import os
import warnings
from typing import Any, List, Tuple

import pytest
from _pytest.fixtures import FixtureRequest
from click.testing import CliRunner, Result

from landshark.scripts import cli, extractors, importers, skcli

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ImportWarning)


@pytest.fixture(scope="module")
def data_loc(request: FixtureRequest) -> Tuple[str, str, str, str, str]:
    """Return the directory of the currently running test script."""
    test_dir = request.fspath.join("..")
    data_dir = os.path.join(test_dir, "data")
    target_dir = os.path.join(data_dir, "targets")
    cat_dir = os.path.join(data_dir, "categorical")
    con_dir = os.path.join(data_dir, "continuous")
    model_dir = os.path.abspath(
        os.path.join(test_dir, "..", "configs"))
    result_dir = os.path.abspath(
        os.path.join(test_dir, "..", "test_output", "pipeline"))
    try:
        os.makedirs(result_dir)
    except FileExistsError:
        pass

    return con_dir, cat_dir, target_dir, model_dir, result_dir


class LandsharkCliRunner(CliRunner):
    """Wrap click.CliRunner to execute Landshark commands."""

    cli_fns = {
        "landshark": cli.cli,
        "landshark-import": importers.cli,
        "landshark-extract": extractors.cli,
        "skshark": skcli.cli,
    }

    def run(self, cmd: List[Any]) -> Result:
        """Execute CLI command using click CliRunner and assert success."""
        assert len(cmd) > 1
        cmd_str = [str(k) for k in cmd]
        print("Running command: {}".format(" ".join(cmd_str)))
        fn = self.cli_fns[cmd_str[0]]
        args = cmd_str[1:]
        result = self.invoke(fn, args)
        assert result.exit_code == 0
        return result


@pytest.fixture(scope="module")
def runner() -> LandsharkCliRunner:
    """CLI runner."""
    return LandsharkCliRunner()

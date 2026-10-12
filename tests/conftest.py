"""Shared fixtures; tests marked `live` drive a real Chrome for Testing (run them with `pytest -m live`)."""

import os
import shutil

import pytest


def _executable(env_var, default):
    path = os.environ.get(env_var) or shutil.which(default)
    return path if path and os.access(path, os.X_OK) else None


@pytest.fixture(scope="session")
def browser_paths():
    """(browser, driver) from QWOP_BROWSER / QWOP_DRIVER or PATH; skips the test if either is missing (fails with QWOP_REQUIRE_LIVE=1)."""
    browser = _executable("QWOP_BROWSER", "chrome_for_testing")
    driver = _executable("QWOP_DRIVER", "chromedriver_for_testing")
    if browser is None or driver is None:
        reason = "Chrome for Testing and chromedriver not found (set QWOP_BROWSER / QWOP_DRIVER)"
        if os.environ.get("QWOP_REQUIRE_LIVE") == "1":
            pytest.fail(reason)
        pytest.skip(reason)
    return browser, driver


@pytest.fixture
def make_env(browser_paths):
    """Factory for headless QwopEnv instances, all closed at teardown."""
    from qwop_gym_leaner import QwopEnv

    envs = []

    def factory(**kwargs):
        kwargs.setdefault("game_in_browser", False)
        env = QwopEnv(browser=browser_paths[0], driver=browser_paths[1], **kwargs)
        envs.append(env)
        return env

    yield factory
    for env in envs:
        env.close()

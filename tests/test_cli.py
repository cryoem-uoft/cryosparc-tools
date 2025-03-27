import subprocess
from argparse import Namespace
from pathlib import Path
from unittest import mock

import pytest

from cryosparc import auth, cli
from cryosparc.tools import CryoSPARC


@pytest.fixture
def user_config_path(monkeypatch, tmp_path):
    monkeypatch.setattr(auth, "user_config_path", mock.Mock(return_value=tmp_path))
    return tmp_path


@pytest.fixture
def mock_auth_path(mock_api_client_class, user_config_path: Path):
    cli.login(
        Namespace(
            url="https://cryosparc.example.com", email="structura@example.com", password="password123", expires=None
        )
    )
    return user_config_path / "cryosparc-tools" / "auth.json"


def test_cli_login(mock_api_client_class, mock_auth_path):
    assert mock_auth_path.is_file()
    mock_api_client_class.login.assert_called_once()


def test_cli_login_auth(mock_user, mock_api_client_class, mock_auth_path):
    cs = CryoSPARC("https://cryosparc.example.com", email="structura@example.com", host=None, base_port=None)
    mock_api_client_class.__call__.assert_called_with(auth="abc123")  # called with token
    assert cs.user == mock_user
    assert cs.test_connection()


def test_has_cli_help():
    output = subprocess.check_output(["python", "-m", "cryosparc.tools", "--help"])
    assert output.startswith(b"usage: cryosparc.tools")


def test_has_cli_login_help():
    output = subprocess.check_output(["python", "-m", "cryosparc.tools", "login", "--help"])
    assert output.startswith(b"usage: cryosparc.tools")

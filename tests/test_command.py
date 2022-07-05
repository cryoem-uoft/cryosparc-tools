import pytest
from cryosparc.command import CommandClient


@pytest.fixture
def cli(command_core):
    return CommandClient()


def test_hello(cli):
    assert cli.hello_world() == {"hello": "world"}

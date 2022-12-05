from cryosparc.tools import CryoSPARC


def test_hello(cs: CryoSPARC):
    assert cs.cli.hello_world() == {"hello": "world"}  # type: ignore

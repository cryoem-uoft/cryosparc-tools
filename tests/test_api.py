from cryosparc.tools import CryoSPARC


def test_health(cs: CryoSPARC):
    assert cs.api.health() == "OK"

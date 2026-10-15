from piper_quickstart import preflight


def test_preflight_fails_cleanly_on_missing_can(capsys):
    code = preflight.main(["--can-port", "can_does_not_exist", "--skip-camera", "--skip-display"])
    out = capsys.readouterr().out
    assert code == 1
    assert "does not exist" in out and "can_activate.sh" in out


def test_preflight_all_skipped_passes():
    assert preflight.main(["--skip-arm", "--skip-camera", "--skip-display"]) == 0

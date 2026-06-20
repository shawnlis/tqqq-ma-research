from research import cli


def test_run_existing_regime_passes_date_arguments(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_run_regime_search(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli, "run_regime_search", fake_run_regime_search)

    assert (
        cli.main(
            [
                "run-existing-regime",
                "--output-dir",
                str(tmp_path / "regime"),
                "--start-date",
                "2020-01-01",
                "--end-date",
                "2020-12-31",
            ]
        )
        == 0
    )

    assert captured["output_dir"] == str(tmp_path / "regime")
    assert captured["start_date"] == "2020-01-01"
    assert captured["end_date"] == "2020-12-31"

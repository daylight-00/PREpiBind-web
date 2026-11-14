"""The hard limits: an epitope of more than MAX_EPITOPE_LENGTH residues and an HLA chain of more than MAX_CHAIN_LENGTH residues are refused, by the engine, before anything is computed."""
import engine
import pandas as pd
import pytest


def test_an_epitope_at_the_limit_is_accepted():
    engine.check_request(["A" * engine.MAX_EPITOPE_LENGTH])


def test_an_epitope_over_the_limit_is_refused_with_its_length():
    with pytest.raises(engine.RequestRefused, match=str(engine.MAX_EPITOPE_LENGTH + 1)):
        engine.check_request(["A" * (engine.MAX_EPITOPE_LENGTH + 1)])


def test_a_chain_at_the_limit_is_accepted_and_one_over_is_refused():
    engine.check_request([], ["A" * engine.MAX_CHAIN_LENGTH])
    with pytest.raises(engine.RequestRefused, match="chain"):
        engine.check_request([], ["A" * (engine.MAX_CHAIN_LENGTH + 1)])


def test_a_refusal_is_a_value_error_so_callers_need_no_new_handler():
    assert issubclass(engine.RequestRefused, ValueError)


def test_the_limits_are_read_when_called_so_a_benchmark_can_raise_them(monkeypatch):
    monkeypatch.setattr(engine, "MAX_EPITOPE_LENGTH", 64)
    engine.check_request(["A" * 64])
    with pytest.raises(engine.RequestRefused):
        engine.check_request(["A" * 65])


def test_the_limit_is_the_one_the_measurements_support():
    """20: every head's AUC steps down between 20 and 21 residues (261115_img1_web_caps_scan_custom/length_auc*.json); the chain limit rounds the catalogue's longest chain (285) up."""
    assert engine.MAX_EPITOPE_LENGTH == 20
    assert engine.MAX_CHAIN_LENGTH == 300


def refuse_any_work(*args, **kwargs):
    raise AssertionError("the request must be refused before a path is chosen")


def test_run_refuses_before_any_work(bare_engine, frame):
    bare_engine._run_fast = bare_engine._run = refuse_any_work
    long = frame.copy()
    long.loc[1, "Epitope"] = "A" * (engine.MAX_EPITOPE_LENGTH + 1)
    with pytest.raises(engine.RequestRefused):
        bare_engine.run(long, "Qualitative")


def test_one_long_epitope_refuses_the_whole_request_not_just_its_row(bare_engine, frame):
    bare_engine._run_fast = bare_engine._run = refuse_any_work
    frame.loc[2, "Epitope"] = "A" * 40
    with pytest.raises(engine.RequestRefused, match="1 epitope"):
        bare_engine.run(frame, "Qualitative", mode="exact")


def test_run_custom_refuses_long_epitopes_and_long_chains(bare_engine):
    bare_engine._run = refuse_any_work
    long_epitope = pd.DataFrame({"MHC": ["AAAA_BBBB"], "Epitope": ["A" * 21]})
    long_chain = pd.DataFrame({"MHC": ["A" * 301 + "_BBBB"], "Epitope": ["A" * 15]})
    with pytest.raises(engine.RequestRefused, match="epitope"):
        bare_engine.run_custom(long_epitope, "Qualitative")
    with pytest.raises(engine.RequestRefused, match="chain"):
        bare_engine.run_custom(long_chain, "Qualitative")


def test_rows_without_an_epitope_are_ignored_by_the_check(bare_engine, frame):
    frame.loc[0, "Epitope"] = None
    bare_engine._run_fast = lambda *a, **k: "fast result"
    assert bare_engine.run(frame, "Qualitative") == "fast result"

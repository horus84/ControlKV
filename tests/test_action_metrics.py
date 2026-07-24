import pytest
from controlkv.metrics.actions import (
    compare_single_action,
    evaluate_ground_truth_correctness,
    evaluate_agreement_against_full_cache
)

def test_compare_single_action_exact_match():
    pred = {"name": "get_weather", "arguments": {"city": "Tokyo"}}
    ref = {"name": "get_weather", "arguments": {"city": "Tokyo"}}
    res = compare_single_action(pred, ref)
    assert res["tool_name_match"] is True
    assert res["args_exact_match"] is True
    assert res["field_accuracy"] == 1.0
    assert res["complete_action_match"] is True

def test_compare_single_action_mismatch():
    pred = {"name": "get_weather", "arguments": {"city": "Osaka"}}
    ref = {"name": "get_weather", "arguments": {"city": "Tokyo"}}
    res = compare_single_action(pred, ref)
    assert res["tool_name_match"] is True
    assert res["args_exact_match"] is False
    assert res["field_accuracy"] == 0.0
    assert res["complete_action_match"] is False

def test_agreement_against_full_cache():
    cand = [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]
    full = [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]
    res = evaluate_agreement_against_full_cache(cand, full)
    assert res["tool_name_agreement"] is True
    assert res["arg_exact_agreement"] is True
    assert res["complete_action_agreement"] is True

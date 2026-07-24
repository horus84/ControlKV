import pytest
from controlkv.parsing.tool_calls import extract_tool_calls

def test_extract_qwen_tool_call():
    text = '<tool_call> {"name": "get_weather", "arguments": {"city": "Tokyo", "units": "celsius"}} </tool_call>'
    valid, calls = extract_tool_calls(text)
    assert valid is True
    assert len(calls) == 1
    assert calls[0]["name"] == "get_weather"
    assert calls[0]["arguments"] == {"city": "Tokyo", "units": "celsius"}

def test_extract_markdown_json_array():
    text = '```json\n[{"name": "search_flights", "arguments": {"origin": "NYC", "destination": "London"}}]\n```'
    valid, calls = extract_tool_calls(text)
    assert valid is True
    assert len(calls) == 1
    assert calls[0]["name"] == "search_flights"
    assert calls[0]["arguments"] == {"origin": "NYC", "destination": "London"}

def test_invalid_tool_call():
    text = "Here is the weather: it is sunny in Tokyo today!"
    valid, calls = extract_tool_calls(text)
    assert valid is False
    assert len(calls) == 0

import json
import runpy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from aiops_pipeline import load_data, run_pipeline
from anomaly_detector import AnomalyDetector
from calculations import area_of_circle, get_nth_fibonacci
from event_producer import EventProducer
from event_topic import EventTopic


def test_calculations_reject_negative_values():
    with pytest.raises(ValueError):
        area_of_circle(-1)

    with pytest.raises(ValueError):
        get_nth_fibonacci(-1)


def test_calculations_compute_recursive_branch():
    assert get_nth_fibonacci(10) == 55


def test_detector_reports_each_anomaly_reason():
    record = {
        "timestamp": "2026-09-20T10:00:00",
        "service": "payment-service",
        "response_time_ms": 501,
        "cpu_percent": 81,
        "memory_percent": 81,
        "log_level": "WARNING",
    }

    event = AnomalyDetector().detect(record)

    assert event["reasons"] == [
        "High response time",
        "High CPU utilization",
        "High memory utilization",
        "Error log detected",
    ]
    assert event["source"] == record


def test_producer_rejects_empty_event():
    topic = EventTopic("anomaly-events")
    assert EventProducer(topic).publish(None) is False
    assert topic.get_messages() == []


def test_topic_clear_and_message_copy():
    topic = EventTopic("anomaly-events")
    event = {"type": "ANOMALY"}
    topic.publish(event)

    messages = topic.get_messages()
    messages.clear()
    assert topic.get_messages() == [event]

    topic.clear()
    assert topic.get_messages() == []


def test_pipeline_loads_data_and_consumes_anomalies(tmp_path):
    records = [
        {
            "timestamp": "2026-09-20T10:00:00",
            "service": "payment-service",
            "response_time_ms": 100,
            "cpu_percent": 40,
            "memory_percent": 50,
            "log_level": "INFO",
        },
        {
            "timestamp": "2026-09-20T10:01:00",
            "service": "payment-service",
            "response_time_ms": 700,
            "cpu_percent": 40,
            "memory_percent": 50,
            "log_level": "ERROR",
        },
    ]
    data_file = tmp_path / "service_data.json"
    data_file.write_text(json.dumps(records), encoding="utf-8")

    assert load_data(data_file) == records
    result = run_pipeline(data_file)

    assert result["records_processed"] == 2
    assert len(result["anomalies_detected"]) == 1
    assert result["events_consumed"] == []


def test_pipeline_script_prints_summary(capsys):
    runpy.run_module("aiops_pipeline", run_name="__main__")
    output = capsys.readouterr().out

    assert "AIOps Pipeline Result" in output
    assert "Records processed: 10" in output
    assert "Anomalies detected: 2" in output

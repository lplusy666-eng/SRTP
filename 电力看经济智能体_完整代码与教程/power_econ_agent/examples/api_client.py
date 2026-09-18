from __future__ import annotations

import argparse
import json

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description="Power Economy Agent API client")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url, timeout=120.0) as client:
        health = client.get("/health")
        health.raise_for_status()
        print("HEALTH")
        print(json.dumps(health.json(), ensure_ascii=False, indent=2))

        events_response = client.get(
            "/v1/anomalies",
            params={"start": "2024-05-01", "end": "2024-05-10", "limit": 3},
        )
        events_response.raise_for_status()
        events = events_response.json()
        print(f"\nEVENTS: {len(events)}")

        if events:
            event_id = events[0]["event_id"]
            diagnosis = client.get(f"/v1/diagnoses/{event_id}")
            diagnosis.raise_for_status()
            print("\nTOP DIAGNOSIS")
            print(json.dumps(diagnosis.json(), ensure_ascii=False, indent=2))

        qa = client.post(
            "/v1/qa",
            json={
                "question": "最近异常主要由什么造成，能否直接说明经济走弱？",
                "session_id": "example-client",
            },
        )
        qa.raise_for_status()
        print("\nQA")
        print(json.dumps(qa.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

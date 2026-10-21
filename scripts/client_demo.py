"""
Client demonstration script.

Shows how an external client consumes the REST API:
1. Verifies service health (/health)
2. Sends a single exercise prediction (/predict)
3. Sends a batch prediction (/predict/batch)
4. Displays formatted output
"""

import json
import requests
import sys

BASE_URL = "http://127.0.0.1:8000"


def main():
    print("=" * 60)
    print("Maritime Navigation Predict API - Client Demonstration")
    print("=" * 60)

    # 1. Health Check
    print("\n[1] Checking API Health...")
    try:
        health_resp = requests.get(f"{BASE_URL}/health", timeout=3)
        print(f"Status: {health_resp.status_code}")
        print(f"Payload: {health_resp.json()}")
        if health_resp.status_code != 200 or not health_resp.json().get("model_loaded"):
            print("Server is not healthy or model is not loaded!")
            sys.exit(1)
    except Exception as e:
        print(f"Failed to connect to API at {BASE_URL}: {e}")
        print("Please ensure the FastAPI server is running.")
        sys.exit(1)

    # 2. Single Exercise Prediction
    print("\n[2] Requesting Single Prediction (/predict)...")
    single_payload = {
        "situations": ["LOW", "MEDIUM", "HIGH", "MEDIUM", "MEDIUM", "LOW"]
    }
    print(f"Request: {json.dumps(single_payload)}")
    pred_resp = requests.post(f"{BASE_URL}/predict", json=single_payload, timeout=5)
    print(f"Status: {pred_resp.status_code}")
    print(f"Response: {json.dumps(pred_resp.json(), indent=2)}")

    # 3. Batch Prediction
    print("\n[3] Requesting Batch Prediction (/predict/batch)...")
    batch_payload = {
        "exercises": {
            "exercise_101": ["LOW", "LOW", "MEDIUM"],
            "exercise_102": ["HIGH", "HIGH", "HIGH", "MEDIUM"],
            "exercise_103": ["LOW", "MEDIUM", "HIGH", "LOW"]
        }
    }
    print(f"Request: {json.dumps(batch_payload, indent=2)}")
    batch_resp = requests.post(f"{BASE_URL}/predict/batch", json=batch_payload, timeout=5)
    print(f"Status: {batch_resp.status_code}")
    print(f"Response: {json.dumps(batch_resp.json(), indent=2)}")

    print("\n" + "=" * 60)
    print("Client demonstration finished successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

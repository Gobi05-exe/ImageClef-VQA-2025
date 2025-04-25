import requests
import json
import time

DATASET = "MBZUAI/EXAMS-V"
CONFIG = "default"
SPLIT = "test"
MAX_LENGTH = 100
TOTAL_ROWS = 3565  # You now know the exact value
OUTPUT_FILE = "exams_v_test_full.json"
MAX_RETRIES = 10

BASE_URL = "https://datasets-server.huggingface.co/rows"
PARAMS_TEMPLATE = {
    "dataset": DATASET,
    "config": CONFIG,
    "split": SPLIT,
    "offset": 0,
    "length": MAX_LENGTH
}

def fetch_rows(offset):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            params = PARAMS_TEMPLATE.copy()
            params["offset"] = offset
            response = requests.get(BASE_URL, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"⚠️  Error {response.status_code} at offset {offset}, retrying...")
        except requests.RequestException as e:
            print(f"⚠️  Request failed: {e}")
        retries += 1
        time.sleep(2 ** retries)  # exponential backoff
    raise Exception(f"❌ Failed to fetch offset {offset} after {MAX_RETRIES} retries.")

def main():
    all_rows = []
    offset = 0
    features = None

    print("🚀 Starting download of test split...")

    while offset < TOTAL_ROWS:
        print(f"Fetching rows {offset} to {min(offset + MAX_LENGTH - 1, TOTAL_ROWS - 1)}...")
        data = fetch_rows(offset)
        if not features:
            features = data.get("features", [])
        rows = data.get("rows", [])
        if not rows:
            print(f"⚠️  No rows returned at offset {offset}. Stopping.")
            break
        all_rows.extend(rows)
        offset += MAX_LENGTH
        time.sleep(0.25)  # polite wait

    print(f"✅ Downloaded {len(all_rows)} rows.")

    final_data = {
        "features": features,
        "rows": all_rows
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2)

    print(f"📦 Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

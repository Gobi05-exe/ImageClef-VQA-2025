"""
Download and save the 'test' split of the MBZUAI/EXAMS-V dataset from Hugging Face.

This script connects to the Hugging Face datasets server, downloads all rows of
the 'test' split in batches, and saves the complete dataset locally as a JSON file.

Features:
- Automatic handling of API request retries with exponential backoff.
- Downloads data in manageable chunks to handle large datasets.
- Saves the final dataset with metadata ('features') and all rows.

Usage:
    python3 Extracter.py
    
"""

import requests
import json
import time

# Constants for dataset fetching
DATASET = "MBZUAI/EXAMS-V"
CONFIG = "default"
SPLIT = "test"
MAX_LENGTH = 100             # Maximum number of rows per request
TOTAL_ROWS = 3565            # Total number of rows in the split
OUTPUT_FILE = "exams_v_test_full.json"  # Output file to save the full dataset
MAX_RETRIES = 10             # Maximum number of retries for failed requests

# Base URL for the Hugging Face datasets server
BASE_URL = "https://datasets-server.huggingface.co/rows"

# Template for API request parameters
PARAMS_TEMPLATE = {
    "dataset": DATASET,
    "config": CONFIG,
    "split": SPLIT,
    "offset": 0,
    "length": MAX_LENGTH
}

def fetch_rows(offset):
    """
    Fetch a batch of dataset rows starting from the specified offset.

    Retries the request up to MAX_RETRIES times in case of connection or server issues,
    with exponential backoff between retries.

    Args:
        offset (int): The starting index from which to fetch rows.

    Returns:
        dict: Parsed JSON response containing 'features' and 'rows'.

    Raises:
        Exception: If the data could not be fetched after MAX_RETRIES attempts.
    """
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
    """
    Main function to download the full 'test' split of the dataset and
    save it to a local JSON file.

    The function sequentially fetches all rows from the dataset in batches,
    handles retries and errors, and finally writes the combined result
    to OUTPUT_FILE.
    """
    all_rows = []
    offset = 0
    features = None

    print("🚀 Starting download of test split...")

    # Loop over dataset until all rows are fetched
    while offset < TOTAL_ROWS:
        print(f"Fetching rows {offset} to {min(offset + MAX_LENGTH - 1, TOTAL_ROWS - 1)}...")
        data = fetch_rows(offset)
        
        # Save dataset features metadata (only once)
        if not features:
            features = data.get("features", [])
        
        # Extract and collect rows
        rows = data.get("rows", [])
        if not rows:
            print(f"⚠️  No rows returned at offset {offset}. Stopping.")
            break
        all_rows.extend(rows)
        
        offset += MAX_LENGTH
        time.sleep(0.25)  # polite wait to avoid rate-limiting

    print(f"✅ Downloaded {len(all_rows)} rows.")

    # Prepare final dataset dictionary
    final_data = {
        "features": features,
        "rows": all_rows
    }

    # Save dataset to a JSON file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2)

    print(f"📦 Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

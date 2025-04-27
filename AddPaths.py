"""
Download all images from a dataset and update the dataset JSON with local image paths.

This script reads a JSON file containing dataset entries with remote image URLs,
downloads each image locally into a specified directory, updates the dataset
entries to include the local file paths, and saves the updated dataset to a new JSON file.

Features:
- Robust download with basic error handling.
- Progress tracking using tqdm.
- Automatic directory creation for downloaded images.

Usage:
    python3 AddPaths.py

"""

import os
import json
import requests
from tqdm import tqdm

# Input and output file paths
INPUT_FILE = "exams_v_test_full.json"
OUTPUT_FILE = "exams_v_test_full_with_paths.json"

# Directory where images will be saved
IMAGE_DIR = "downloaded_images"

# Ensure the image directory exists
os.makedirs(IMAGE_DIR, exist_ok=True)

def download_image(url, save_path):
    """
    Download an image from a URL and save it to a local path.

    Args:
        url (str): The URL of the image to download.
        save_path (str): The local filesystem path to save the image.

    Returns:
        bool: True if the download was successful, False otherwise.
    """
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            print(f" Failed to download {url}: Status {response.status_code}")
    except Exception as e:
        print(f" Error downloading {url}: {e}")
    return False

def main():
    """
    Main function to download images for each dataset entry and update the dataset JSON.

    Reads the input JSON, downloads each referenced image, adds the local image path
    into the dataset, and writes the modified dataset to the output JSON file.
    """
    # Load the dataset
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Iterate through dataset entries and download images
    for entry in tqdm(data["rows"], desc="Downloading images"):
        row_idx = entry["row_idx"]
        image_info = entry["row"]["image"]
        image_url = image_info["src"]

        # Create a consistent local filename
        image_filename = f"image_{row_idx}.png"
        image_path = os.path.join(IMAGE_DIR, image_filename)

        # Download the image and update the entry
        if download_image(image_url, image_path):
            # Save the relative path in the dataset
            entry["row"]["image"]["path"] = os.path.join(IMAGE_DIR, image_filename)

    # Save the updated dataset
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f" Done! Images saved in '{IMAGE_DIR}' and new JSON in '{OUTPUT_FILE}'")

if __name__ == "__main__":
    main()

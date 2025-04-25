import os
import json
import requests
from tqdm import tqdm

INPUT_FILE = "exams_v_test_full.json"
OUTPUT_FILE = "exams_v_test_full_with_paths.json"
IMAGE_DIR = "downloaded_images"

os.makedirs(IMAGE_DIR, exist_ok=True)

def download_image(url, save_path):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            print(f"❌ Failed to download {url}: Status {response.status_code}")
    except Exception as e:
        print(f"⚠️ Error downloading {url}: {e}")
    return False

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in tqdm(data["rows"], desc="Downloading images"):
        row_idx = entry["row_idx"]
        image_info = entry["row"]["image"]
        image_url = image_info["src"]
        image_filename = f"image_{row_idx}.png"
        image_path = os.path.join(IMAGE_DIR, image_filename)

        if download_image(image_url, image_path):
            # Add relative path to image info
            entry["row"]["image"]["path"] = os.path.join(IMAGE_DIR, image_filename)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Done! Images saved in '{IMAGE_DIR}' and new JSON in '{OUTPUT_FILE}'")

if __name__ == "__main__":
    main()

import os
import shutil

from better_bing_image_downloader import downloader

# Constants
TARGET_WORD = "fork"
HARD_NEGATIVES = ["spoon", "knife", "backscratcher", "ladle", "tong", "whisk"]
OUTPUT_DIR = 'images'
POSITIVE_DIR = os.path.join(OUTPUT_DIR, 'positive')
NEGATIVE_DIR = os.path.join(OUTPUT_DIR, 'negative')
SEARCH_BUFFER_FACTOR = 1.5 # how many more to search for in case of download failure
HARD_PORTION = 0.3
POSITIVE_PORTION = 0.5
DATASET_SIZE = 1000

POS_COUNT = round(DATASET_SIZE * POSITIVE_PORTION)
HARD_COUNT = round(round(DATASET_SIZE * HARD_PORTION) // len(HARD_NEGATIVES)) # count per hard negative

def download_images(query: str, count: int, output_dir: str):
    downloader(
    query=query, 
    limit=round(count * SEARCH_BUFFER_FACTOR),
    output_dir=output_dir,
    adult_filter_off=True, 
    force_replace=False, 
    timeout=50,
    verbose=False
)
    
    subfolder = os.path.join(output_dir, query)
    file_count = 0

    # empty out the evil subfolder
    if os.path.exists(subfolder):
        for filename in os.listdir(subfolder):
            if file_count >= count: # Exit if reached quota
                break

            if filename == '_manifest.json': # dont copy the metadata file
                continue

            # Move file to parent folder
            src_dir = os.path.join(subfolder, filename)
            dst_dir = os.path.join(output_dir, filename)
            shutil.move(src_dir, dst_dir)
            file_count += 1

    # Delete the evil subfolder
    shutil.rmtree(subfolder)

def main():
    # Create directory if not already exists
    os.makedirs(POSITIVE_DIR, exist_ok=True)
    os.makedirs(NEGATIVE_DIR, exist_ok=True)
    # --- Scraping ---

    # 1. Scrape the target word
    print(f"Now downloading: {TARGET_WORD}...")
    download_images(TARGET_WORD, POS_COUNT, POSITIVE_DIR)
    print("Downloaded positives!")

    # 2. Scrape from the hard negative list
    for word in HARD_NEGATIVES:
        print(f"Now downloading: {word}...")
        download_images(word, HARD_COUNT, NEGATIVE_DIR)

    print("Downloaded hard negatives!")

    # Easy negatives to be taken from COCO database

if __name__ == '__main__':
    main()
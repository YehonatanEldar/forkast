import os
import shutil

from better_bing_image_downloader import downloader

# Constants
FORK_QUERIES = [
    "metal dinner fork",
    "stainless steel fork",
    "plastic fork utensil",
    "salad fork plate",
    "dessert fork table",
    "kitchen fork silver"
]
HARD_NEGATIVES = ["spoon", "knife", "ladle", "tong", "whisk"]
OUTPUT_DIR = 'images'
POSITIVE_DIR = os.path.join(OUTPUT_DIR, 'positive')
NEGATIVE_DIR = os.path.join(OUTPUT_DIR, 'negative')
NUM_IMAGES = 200

def download_images(query: str, count: int, output_dir: str):
    """
    Downloads images matching the query
    """
    downloader(
    query=query, 
    limit=count,
    output_dir=output_dir,
    adult_filter_off=True, 
    force_replace=False, 
    timeout=50,
    verbose=False
)
    
    subfolder = os.path.join(output_dir, query)

    # empty out the evil subfolder
    if os.path.exists(subfolder):
        for filename in os.listdir(subfolder):

            if filename == '_manifest.json': # dont copy the metadata file
                continue

            # Move file to parent folder
            src_dir = os.path.join(subfolder, filename)
            dst_dir = os.path.join(output_dir, filename)
            shutil.move(src_dir, dst_dir)

    # Delete the evil subfolder
    shutil.rmtree(subfolder)

def main():
    # Create directory if not already exists
    os.makedirs(POSITIVE_DIR, exist_ok=True)
    os.makedirs(NEGATIVE_DIR, exist_ok=True)
    # --- Scraping ---

    # 1. Scrape the target word
    for query in FORK_QUERIES:
        print(f"Now downloading: {query}...")
        download_images(query, NUM_IMAGES, POSITIVE_DIR)

    print("Downloaded positives!")

    # 2. Scrape from the hard negative list
    for word in HARD_NEGATIVES:
        print(f"Now downloading: {word}...")
        download_images(word, NUM_IMAGES, NEGATIVE_DIR)

    print("Downloaded hard negatives!")

    # Easy negatives to be taken from COCO database

if __name__ == '__main__':
    main()
    dir_path = 'images/positive'
    # List all entries and filter out directories
    file_count = len([item for item in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, item))])

    print(f"Number of files: {file_count}")
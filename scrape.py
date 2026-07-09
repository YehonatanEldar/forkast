import os

import requests
from ddgs import DDGS
from better_bing_image_downloader import downloader
# Constants
TARGET_WORD = "fork"
HARD_NEGATIVES = ["spoon", "knife", "backscratcher", "ladle", "tong", "whisk"]
OUTPUT_DIR = 'images'
SEARCH_BUFFER_FACTOR = 1.5 # how many more to search for in case of download failure
HARD_PORTION = 0.3
POSITIVE_PORTION = 0.5
DATASET_SIZE = 100

POS_COUNT = round(DATASET_SIZE * POSITIVE_PORTION)
HARD_COUNT = round(round(DATASET_SIZE * HARD_PORTION) // len(HARD_NEGATIVES)) # count per hard negative

def get_image_urls(word: str, count: int) -> list[dict]:
    """
    Returns a list of urls to images of the given word
    """
    with DDGS() as ddgs:
        return list(ddgs.images(word, max_results=round(count * SEARCH_BUFFER_FACTOR)))
    

# def download_images(results: list[str], count: int, output_dir: str):
#     """
#     Downloads the given url list to the given folder path
#     """
#     download_count = 0

#     for item in results:
#         if download_count >= count:
#             break

#         url = item['image']

#         try: 
#             response = requests.get(url)
#             if response.status_code == 200:
#                 # Copy the contents to a file
#                 download_count += 1
#                 with open(f"{output_dir}/{download_count}.jpg", "wb") as f:
#                     f.write(response.content)

#                 print(f"Downloaded: {download_count}/{count}")


#         except Exception as e:
#             print(e)

def download_images(query: str, count: int, output_dir: str):
    downloader(
    query=query, 
    limit=count,                  # <--- Set this to whatever number you need (e.g., 60, 100, 200)
    output_dir=output_dir,  # Destination folder
    adult_filter_off=True, 
    force_replace=False, 
    timeout=50,
    verbose=False
)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Create directory if not already exists
    os.makedirs(OUTPUT_DIR + '/positive', exist_ok=True)
    os.makedirs(OUTPUT_DIR + '/negative', exist_ok=True)
    # --- Scraping ---

    # 1. Scrape the target word
    download_images(TARGET_WORD, POS_COUNT, OUTPUT_DIR + '/positive')
    print("Done!")

    # 2. Scrape from the hard negative list
    # for word in HARD_NEGATIVES:
    #     results = get_image_urls(word, HARD_COUNT, OUTPUT_DIR + '/negative')

if __name__ == '__main__':
    main()

from typing import List
from crawl_data.utils import download_page, download_img
from bs4 import BeautifulSoup
import os
import time

URL = 'https://www.drive.com.au/cars-for-sale/'
OUTPUT_DIR = 'crawl_drive'


def get_all_img_urls(page_html) -> List:
    """Get all the url of car images from the page drive.com.au

    Note that the detail here is specific to the page. The class was obtained by
    inspecting the page
    """
    soup = BeautifulSoup(page_html, 'html.parser')
    imgs = soup.find_all(
        'img', class_='image_drive-image__f_jEC listingDetailsCard_drive-cfs__listing-card__img__FjiUX bg-gray-200')

    img_urls = [img['src'] for img in imgs]

    return img_urls


def download_all() -> None:
    max_page = 500  # too lazy to grab this programatically
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for page in range(1, max_page+1):
        page_url = URL
        if page != 1:
            page_url += f'page/{page}/'

        print(f'processing {page_url}')
        page_html = download_page(page_url)

        if page_html is not None:
            page_imgs = get_all_img_urls(page_html)
            print(len(page_imgs))
            for img in page_imgs:
                download_img(img, OUTPUT_DIR)

        time.sleep(1)


if __name__ == "__main__":
    download_all()

from typing import List
from crawl_data.utils import download_page, download_img
from bs4 import BeautifulSoup
import os
import time

URL = 'https://www.sydneycars.com.au/'
OUTPUT_DIR = 'crawl_sydneycars'


def get_all_img_urls(page_html) -> List:
    """Get all the url of car images from the page drive.com.au

    Note that the detail here is specific to the page. The class was obtained by
    inspecting the page
    """

    soup = BeautifulSoup(page_html, 'html.parser')
    imgs = soup.find_all('img', class_='img-fluid')

    img_urls = [img['data-src'] for img in imgs]

    return img_urls


def download_all() -> None:
    max_page = 27  # too lazy to grab this programatically
    os.makedirs('crawl_sydneycars', exist_ok=True)
    for page in range(1, max_page+1):
        page_url = URL
        if page != 1:
            page_url += f'page/{page}/'

        print(f'processing {page_url}')
        page_html = download_page(page_url)

        if page_html is not None:
            page_imgs = get_all_img_urls(page_html)
            for img in page_imgs:
                download_img(img,OUTPUT_DIR)

        time.sleep(1)


if __name__ == "__main__":
    download_all()

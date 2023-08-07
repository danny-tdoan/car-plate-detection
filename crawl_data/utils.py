import os
import requests
from typing import Optional


def download_page(url: str) -> Optional[str]:
    page = requests.get(url)
    if page.status_code == 200:
        return page.text


def download_img(img_url: str, output_dir: str) -> None:
    img_name = img_url.split('/')[-1]
    img = requests.get(img_url)

    if img.status_code == 200:
        open(os.path.join(output_dir, img_name), 'wb').write(img.content)
        print(f'downloaded and saved {img_name}')
    else:
        print(f'failed to download and save {img_name}')

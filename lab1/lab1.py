import os
from threading import Thread
import requests
from datetime import datetime


class DownloadThread(Thread):
    def __init__(self, url, tar_dir):
        Thread.__init__(self)
        self.url = url
        self.tar_dir = tar_dir

    def run(self):
        get_response = requests.get(self.url, stream=True)
        file_name = self.url.split("/")[-1]
        tar_path = os.path.join(self.tar_dir, file_name)
        if os.path.exists(tar_path):
            name, ext = file_name.split(".")
            suf = (
                datetime.utcnow()
                .isoformat()
                .replace(":", "-")
                .replace('-', '_')
            )
            tar_path = os.path.join(self.tar_dir, f"{name}_{suf}.{ext}")
        with open(tar_path, 'wb') as f:
            for chunk in get_response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)


def download_mt(urls, tar_dir):
    threads = []
    for url in urls:
        new_thread = DownloadThread(url, tar_dir)
        threads.append(new_thread)
    if not os.path.exists(tar_dir):
        os.mkdir(tar_dir)
    print(f'Created {len(threads)} threads')
    for thread in threads:
        print('Starting thread for url: ', thread.url)
        thread.start()
    for thread in threads:
        thread.join()
        print('Thread for url: ', thread.url, ' has finished downloading')
    print('All done, results available under: ', tar_dir)


if __name__ == "__main__":
    url_list = ['https://placebear.com/200/300.jpg',
                'https://placebear.com/300/200.jpg',
                'https://placebear.com/400/400.jpg']
    directory = ""
    download_mt(url_list, directory)

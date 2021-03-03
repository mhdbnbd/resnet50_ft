import os
import shutil
from pathlib import Path
import requests


def download_file_from_google_drive(id, destination):
    # Source: https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 200000

    print("Downloading")
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def download_and_extract_office31(target_path):

    tmp_path = Path("./tmp.tar.gz").absolute()
    download_file_from_google_drive(
        id="0B4IapRTv9pJ1WGZVd1VDMmhwdlE", destination=str(tmp_path),
    )

    os.mkdir(target_path)

    print("Unpacking")
    shutil.unpack_archive(str(tmp_path), target_path)

    print("Cleaning up")
    tmp_path.unlink()

    print("Done")


def build_dataset(ds_path=os.path.join("Office31"), split_path=os.path.join("data", "splits_structure")):

    if not os.path.exists(split_path):
        print("splits_structure files not found !")

    else:
        if not os.path.exists(ds_path):
            download_and_extract_office31(ds_path)
            assert os.path.exists(ds_path)

        domains = os.listdir(ds_path)
        splits = ["_half", "_half2"]

        # retrieve classes from folder names in the original dataset
        classes_path = os.path.join(ds_path, "amazon/images")
        classes = os.listdir(classes_path)

        # create split directories and classes direrctories within
        splits_paths = [os.path.join(ds_path, domain + split, "images", _class) for domain in domains for split in
                        splits
                        for _class in classes]
        if len(os.listdir(ds_path)) == 3:
            for path in splits_paths:
                if not os.path.exists(path):
                    os.makedirs(path)
            print("{} directories ({} classes in {} domains *{} splits) created".format(len(splits_paths), len(classes),
                                                                                        len(domains), len(splits)))

            count = 0
            for domain in domains:
                for split in splits:
                    path_file = os.path.join(split_path, domain + split, "images.txt")
                    f = open(path_file, 'r')
                    for line in f:
                        image_path = os.path.join(ds_path, domain + split, "images", line[2:-1])
                        source = os.path.join(ds_path, domain, "images", line[2:-1])
                        destination = image_path
                        shutil.copy2(source, destination)
                        count += 1

            print(count, "files copied in", [domain + split for domain in domains for split in splits])

        else:
            print('Files already created')

import gdown
import zipfile

if __name__ == "__main__":
    file_id = "1UlmIdSCohybVoz1cEHp4uohenuH95lrk"
    url = "https://drive.google.com/uc?id=" + file_id
    output = "preprocessed.zip"
    gdown.download(url, output, quiet=False)

    with zipfile.ZipFile(output, "r") as zip_ref:
        zip_ref.extractall(".")

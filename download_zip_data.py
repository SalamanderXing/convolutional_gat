import gdown
import zipfile

if __name__ == "__main__":
#    file_id = "1UlmIdSCohybVoz1cEHp4uohenuH95lrk"
#    file_id = "1acc0jYSLkoQVcse42BOxfEKHvdC0fxfc"
#    file_id = "1W0O_-42eM909kDc-rKHzijZO1DI428Jz"
    file_id = "1acc0jYSLkoQVcse42BOxfEKHvdC0fxfc"
    url = "https://drive.google.com/uc?id=" + file_id
    output = "preprocessed.zip"
    gdown.download(url, output, quiet=False)

    with zipfile.ZipFile(output, "r") as zip_ref:
        zip_ref.extractall(".")

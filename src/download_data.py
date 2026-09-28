import os
import urllib.request

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Full Uber 2021 SEC 10-K Filing (~100+ pages of financial statements, risk factors, and operations)
DOC_URL = "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/10k/uber_2021.pdf"
LOCAL_FILE = os.path.join(DATA_DIR, "annual_report.pdf")


def download_full_10k():
    print(f"Downloading comprehensive enterprise 10-K filing to {LOCAL_FILE}...")
    # Add User-Agent header so GitHub / SEC endpoints don't reject or truncate the stream
    req = urllib.request.Request(
        DOC_URL,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    with urllib.request.urlopen(req) as response, open(LOCAL_FILE, "wb") as out_file:
        data = response.read()
        out_file.write(data)

    file_size_kb = os.path.getsize(LOCAL_FILE) / 1024
    print(f"Downloaded successfully: {file_size_kb:.2f} KB")


if __name__ == "__main__":
    download_full_10k()

"""Download the public CC BY 4.0 dataset; never commit raw transactions."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with urlopen(URL, timeout=120) as response:
        payload = response.read()
    with ZipFile(io.BytesIO(payload)) as archive:
        names = [n for n in archive.namelist() if n.endswith(".xlsx")]
        if len(names) != 1:
            raise ValueError("Expected one Excel file in the UCI archive")
        excel = archive.read(names[0])
    target = args.out / "online-retail.xlsx"
    target.write_bytes(excel)
    metadata = {"url": URL, "sha256": hashlib.sha256(excel).hexdigest(),
                "citation": "Chen, D. (2015). Online Retail. UCI. DOI:10.24432/C5BW33",
                "license": "CC BY 4.0"}
    (args.out / "source.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()

from hashlib import sha256
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zipfile import ZipFile
import json
import os


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
EXTRACTED = ROOT / "data" / "extracted" / "california"
ARCHIVE_URL = "https://data.nlr.gov/system/files/205/California-1679428758.zip"
LICENSE_URL = "https://data.nlr.gov/node/205/license"
PVWATTS_URL = "https://developer.nlr.gov/api/pvwatts/v8.json"
ARCHIVE_SHA256 = "d471d50c91e94ca4b52f626e9677445062910bf6a208acbfaed00c01122b3e0e"


def download(url: str, destination: Path, referer: str | None = None) -> None:
    headers = {"User-Agent": "dc-microgrid-reproducibility/1.0"}
    if referer:
        headers["Referer"] = referer
    with urlopen(Request(url, headers=headers), timeout=180) as response:
        destination.write_bytes(response.read())


RAW.mkdir(parents=True, exist_ok=True)
EXTRACTED.mkdir(parents=True, exist_ok=True)
archive = RAW / "California.zip"
download(ARCHIVE_URL, archive, "https://data.nlr.gov/submissions/205")
if sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
    raise RuntimeError("California archive checksum mismatch")
download(LICENSE_URL, RAW / "dataset205_license.html")

parameters = {
    "api_key": os.environ.get("NREL_API_KEY", "DEMO_KEY"),
    "azimuth": 180,
    "system_capacity": 1,
    "losses": 14,
    "array_type": 1,
    "module_type": 1,
    "tilt": 20,
    "lat": 34.1478,
    "lon": -118.1445,
    "timeframe": "hourly",
    "dc_ac_ratio": 1.2,
    "gcr": 0.4,
    "inv_eff": 96,
    "radius": 0,
    "dataset": "nsrdb",
}
pv_destination = RAW / "pvwatts_pasadena_1kw.json"
download(f"{PVWATTS_URL}?{urlencode(parameters)}", pv_destination)
pv = json.loads(pv_destination.read_text(encoding="utf-8"))
if pv.get("errors") or any(len(pv["outputs"][key]) != 8760 for key in ("dc", "ac", "tamb")):
    raise RuntimeError("PVWatts response failed the 8760 hour validation")

with ZipFile(archive) as source:
    source.extractall(EXTRACTED)

print(json.dumps({"archive_sha256": ARCHIVE_SHA256, "pvwatts_version": pv["version"], "hours": len(pv["outputs"]["dc"])}, indent=2))


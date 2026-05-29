from pathlib import Path
import requests

Path("data/raw").mkdir(parents=True, exist_ok=True)

url = "https://www.daggegevens.knmi.nl/klimatologie/daggegevens"

payload = {
    "start": "20150101",
    "end": "20241231",
    "vars": "DDVEC:FHVEC:FG:FHX:FXX",
    "stns": "ALL",
    "fmt": "csv",
}

response = requests.post(url, data=payload, timeout=120)
response.raise_for_status()

out = Path("data/raw/knmi_daily_wind_2015_2024.csv")
out.write_text(response.text, encoding="utf-8")

print(f"Saved: {out}")
print(response.text[:1000])
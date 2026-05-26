import requests
import xarray as xr
import os
import math

# Константы из документации KNMI
API_KEY = "eyJvcmciOiI1ZTU1NGUxOTI3NGE5NjAwMDEyYTNlYjEiLCJpZCI6ImVlNDFjMWI0MjlkODQ2MThiNWI4ZDViZDAyMTM2YTM3IiwiaCI6Im11cm11cjEyOCJ9"
BASE_URL = "https://api.dataplatform.knmi.nl/open-data/v1/datasets/knmi23_user_friendly_racmo/versions/3.0/files"
headers = {"Authorization": API_KEY}


def download_file(filename):
    if os.path.exists(filename):
        print(f"Файл {filename} уже существует, пропускаю скачивание.")
        return filename

    print(f"Скачиваю {filename}...")
    url_res = requests.get(f"{BASE_URL}/{filename}/url", headers=headers)
    if url_res.status_code != 200:
        print(f"Ошибка при получении ссылки для {filename}: {url_res.text}")
        return None

    download_url = url_res.json().get("temporaryDownloadUrl")
    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return filename


# Имена файлов для сценария 2050 (из списка доступных на платформе)
file_uas = "uas_Hd_2050_interp.nc"
file_vas = "vas_Hd_2050_interp.nc"

# 1. Скачиваем
path_uas = download_file(file_uas)
path_vas = download_file(file_vas)

if path_uas and path_vas:
    # 2. Открываем данные
    ds_uas = xr.open_dataset(path_uas)
    ds_vas = xr.open_dataset(path_vas)

    # 3. Расчет скорости ветра (Vector Magnitude)
    # Формула: sqrt(uas^2 + vas^2)
    wind_speed = (ds_uas['uas'] ** 2 + ds_vas['vas'] ** 2) ** 0.5
    wind_speed.name = "wind_speed"

    # 4. Печать структуры и первых данных
    print("\n--- Итоговый расчет скорости ветра ---")
    print(wind_speed)

    # Берем первый ансамбль (ens=1) и первую дату, выводим срез 5x5 клеток
    sample = wind_speed.isel(time=0, ens=0).sel(lat=slice(52, 52.5), lon=slice(4.5, 5.0))
    print("\n--- Пример данных (Скорость ветра м/с) для участка Нидерландов ---")
    print(sample.to_pandas())

    # Закрываем файлы
    ds_uas.close()
    ds_vas.close()
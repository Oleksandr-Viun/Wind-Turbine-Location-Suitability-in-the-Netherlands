import requests
import pandas as pd
import xarray as xr
import os

# Используем анонимный ключ из документации (действует до июля 2026)
API_KEY = "eyJvcmciOiI1ZTU1NGUxOTI3NGE5NjAwMDEyYTNlYjEiLCJpZCI6ImVlNDFjMWI0MjlkODQ2MThiNWI4ZDViZDAyMTM2YTM3IiwiaCI6Im11cm11cjEyOCJ9"
BASE_URL = "https://api.dataplatform.knmi.nl/open-data/v1/datasets/etmaalgegevensKNMIstations/versions/1/files"
headers = {"Authorization": API_KEY}

def explore_etmaal_data():
    # 1. Запрашиваем список файлов (берём первые 3 для проверки)
    print("Запрашиваю список файлов из API...")
    response = requests.get(BASE_URL, headers=headers, params={"maxKeys": 3})
    
    if response.status_code != 200:
        print(f"Ошибка доступа к API: {response.status_code} - {response.text}")
        return

    files = response.json().get("files", [])
    if not files:
        print("Список файлов пуст.")
        return

    # Выводим список доступных файлов для информации
    print("\nДоступные файлы на сервере:")
    for f in files:
        print(f"- {f['filename']} ({f['size']} байт)")

    # Берем самый первый файл для теста
    filename = files[0].get("filename")
    print(f"\nВыбираю для скачивания: {filename}")

    # 2. Получаем временную URL-ссылку на скачивание этого файла
    url_endpoint = f"{BASE_URL}/{filename}/url"
    url_response = requests.get(url_endpoint, headers=headers)
    download_url = url_response.json().get("temporaryDownloadUrl")

    # 3. Скачиваем сам файл
    print("Скачиваю файл на компьютер...")
    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Файл {filename} успешно скачан!")

    # 4. Пробуем открыть и прочитать, что внутри
    print("\n--- Анализ содержимого файла ---")
    
    # Если файл NetCDF (.nc)
    if filename.endswith('.nc'):
        ds = xr.open_dataset(filename)
        print(ds)
        ds.close()
    
    # Если файл текстовый/CSV (.txt, .csv)
    elif filename.endswith(('.txt', '.csv')):
        # Читаем первые 20 строк как текст, чтобы понять структуру (там часто длинный заголовок от KNMI)
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in range(20):
                print(f.readline().strip())
                
    else:
        print(f"Неизвестный формат файла. Размер на диске: {os.path.getsize(filename)} байт")

# Запускаем скрипт
explore_etmaal_data()
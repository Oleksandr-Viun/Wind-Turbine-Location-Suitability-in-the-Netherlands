import requests
import os

# Твой анонимный ключ (или вставь свой зарегистрированный, если этот истечет)
API_KEY = "eyJvcmciOiI1ZTU1NGUxOTI3NGE5NjAwMDEyYTNlYjEiLCJpZCI6ImVlNDFjMWI0MjlkODQ2MThiNWI4ZDViZDAyMTM2YTM3IiwiaCI6Im11cm11cjEyOCJ9"
DATASET_NAME = "etmaalgegevensKNMIstations"
DATASET_VERSION = "1"
BASE_URL = f"https://api.dataplatform.knmi.nl/open-data/v1/datasets/{DATASET_NAME}/versions/{DATASET_VERSION}/files"
headers = {"Authorization": API_KEY}

def download_all_etmaal_data():
    print("Запрашиваю список всех файлов...")
    # Берем с запасом (maxKeys=500), чтобы точно захватить всё
    response = requests.get(BASE_URL, headers=headers, params={"maxKeys": 500})
    
    if response.status_code != 200:
        print(f"Ошибка API: {response.status_code}")
        return []

    files = response.json().get("files", [])
    print(f"Найдено файлов для скачивания: {len(files)}")

    downloaded_files = []

    for f in files:
        filename = f['filename']
        # Проверяем, не скачали ли мы его уже
        if os.path.exists(filename):
            print(f"Файл {filename} уже существует, пропускаю.")
            downloaded_files.append(filename)
            continue
            
        print(f"Скачиваю {filename} ({f['size'] / 1024 / 1024:.2f} МБ)...")
        url_res = requests.get(f"{BASE_URL}/{filename}/url", headers=headers)
        download_url = url_res.json().get("temporaryDownloadUrl")
        
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(filename, "wb") as file_obj:
                for chunk in r.iter_content(chunk_size=8192):
                    file_obj.write(chunk)
                    
        downloaded_files.append(filename)
        
    print("\nВсе файлы успешно скачаны!")
    return downloaded_files

# Запускаем загрузку
saved_files = download_all_etmaal_data()
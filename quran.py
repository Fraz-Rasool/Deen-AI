import requests
import json
# url  = "https://api.alquran.cloud/v1/quran/en.asad"
url = "https://api.alquran.cloud/v1/quran/quran-uthmani"

try:
    # Stream the response to handle large payload
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    # Read the full content safely
    data = json.loads(response.content.decode("utf-8"))

    # Save to a file
    with open("quran_uthmani.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("Quran data fetched and saved successfully!")

except requests.exceptions.RequestException as e:
    print(f"Error fetching Quran data: {e}")

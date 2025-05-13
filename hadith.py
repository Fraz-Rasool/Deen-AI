import requests
import json
import os

for i in range(1,1620):
    api_url = f'https://hadithapi.com/api/hadiths?page={i}&apiKey=$2y$10$LD4Ap7JAMxQhWDNs6VJqOncSwrtMspXqv8wCnOgYczDZHGuRxvO'

# Set your desired file path
    file_path = f"Sahi Bukhari Ahadith/sahi_bukhari_hadith_page{i}.json"

    try:
        response = requests.get(api_url)
        print("Status Code:", response.status_code)

        response.raise_for_status()  # Stop if status is not 200

        data = response.json()
       # print("Preview of Response:", json.dumps(data, indent=2)[:500])  # First 500 chars

        # Save to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"✅ Hadiths saved successfully to:\n{file_path}")

    except requests.exceptions.RequestException as error:
        print(f"❌ Error during request: {error}")
    except Exception as e:
        print(f"❌ Error during file saving: {e}")

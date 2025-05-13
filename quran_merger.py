import json
import os

# Re-load the files after code state reset
uthmani_path = "quran_uthmani.json"
asad_path = "quran_asad.json"

with open(uthmani_path, "r", encoding="utf-8") as f_ar:
    arabic_quran = json.load(f_ar)["data"]["surahs"]

with open(asad_path, "r", encoding="utf-8") as f_en:
    translation_quran = json.load(f_en)["data"]["surahs"]

# Merge with revelationType included
merged_with_revelation = []

for ar_surah, en_surah in zip(arabic_quran, translation_quran):
    merged_surah = {
        "surah_number": ar_surah["number"],
        "surah_name_ar": ar_surah["name"],
        "surah_name_en": ar_surah["englishName"],
        "revelation_type": ar_surah.get("revelationType", "Unknown"),
        "ayahs": []
    }

    for ar_ayah, en_ayah in zip(ar_surah["ayahs"], en_surah["ayahs"]):
        merged_surah["ayahs"].append({
            "ayah_number": ar_ayah["numberInSurah"],
            "arabic": ar_ayah["text"],
            "translation": en_ayah["text"]
        })

    merged_with_revelation.append(merged_surah)

# Save the final output
final_output_path = "merged_quran_with_revelation.json"
with open(final_output_path, "w", encoding="utf-8") as f_out:
    json.dump(merged_with_revelation, f_out, ensure_ascii=False, indent=2)

final_output_path

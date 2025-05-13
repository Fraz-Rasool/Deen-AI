import csv
import json

final_output_path = "merged_quran_with_revelation.json"
# Load the merged Quran JSON file
with open(final_output_path, "r", encoding="utf-8") as f:
    merged_quran_data = json.load(f)

# Define the CSV file path
csv_output_path = "merged_quran.csv"

# Create and write to the CSV
with open(csv_output_path, "w", encoding="utf-8-sig", newline="") as csvfile:
    writer = csv.writer(csvfile)
    
    # Write the header
    writer.writerow([
        "Revelation Type", "Surah Number", "Surah Name (English)", "Surah Name (Arabic)",
        "Ayah Number", "Ayah Arabic", "Ayah Translation"
    ])
    
    # Write each ayah's data
    for surah in merged_quran_data:
        for ayah in surah["ayahs"]:
            writer.writerow([
                surah["revelation_type"],
                surah["surah_number"],
                surah["surah_name_en"],
                surah["surah_name_ar"],
                ayah["ayah_number"],
                ayah["arabic"],
                ayah["translation"]
            ])

csv_output_path

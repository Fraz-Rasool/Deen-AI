import os
import sys
from dotenv import load_dotenv
from typing import Dict
from quran_helper import user_query  # Quran query function
from sahih_bhukari_helper import user_query_sahi_bukhari  # Bukhari query function
from merger_helper import unified_query  # The summary merger helper
from sahih_muslim_helper import user_query_sahi_muslim  # Muslim query function

# Load environment variables
load_dotenv()

def query_all_sources(question: str) -> Dict[str, str]:
    return {
        "Quran": user_query(question),
        "Sahih Bukhari": user_query_sahi_bukhari(question),
        "Sahih Muslim": user_query_sahi_muslim(question)
    }

def display_results(summary: str, results: Dict[str, str]):
    print("\n" + "="*50)
    print(f"ISLAMIC KNOWLEDGE ASSISTANT - RESULTS")
    print("="*50)

    print("\n[SUMMARY]")
    print("-"*50)
    print(summary)

    for source, response in results.items():
        print(f"\n[{source.upper()} REFERENCES]")
        print("-"*50)
        print(response)

    print("\n" + "="*50)
    print("End of Results")
    print("="*50 + "\n")

def main():
    print("""
    ********************************************
    * ISLAMIC KNOWLEDGE ASSISTANT             *
    * Provides authentic references from:     *
    * 1. The Holy Quran                       *
    * 2. Sahih Bukhari Hadith                 *
    * 3. Sahih Muslim Hadith                  *
    * Also provides a unified summary          *
    ********************************************
    """)

    while True:
        try:
            question = input("\nAsk your Islamic question (or type 'exit' to quit): ").strip()
            if question.lower() in ['exit', 'quit']:
                print("\nMay Allah guide us all to the right path. Assalamu Alaikum!")
                break

            if not question:
                print("Please enter a valid question.")
                continue

            print("\nSearching authentic Islamic sources...")

            # Get merged summary from all sources
            summary = unified_query(question)
            
            # Get individual detailed references
            results = query_all_sources(question)


            display_results(summary, results)

        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            continue

if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY") or not os.getenv("PINECONE_API_KEY"):
        print("❌ Missing GEMINI_API_KEY or PINECONE_API_KEY in .env")
        sys.exit(1)

    main()

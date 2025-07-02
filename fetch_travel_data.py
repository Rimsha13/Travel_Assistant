import requests
from bs4 import BeautifulSoup
import os

def scrape_wikivoyage(country_name):
    formatted_name = country_name.replace(" ", "_")  # handle countries with spaces
    base_url = f"https://en.wikivoyage.org/wiki/{formatted_name}"
    response = requests.get(base_url)

    if response.status_code != 200:
        print(f"[❌] Failed to retrieve page for {country_name}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    text_content = "\n".join([para.get_text(strip=True) for para in paragraphs])
    return text_content

def save_travel_file(country_name, content):
    if not os.path.exists("travel_data"):
        os.makedirs("travel_data")
    filepath = f"travel_data/{country_name.lower().replace(' ', '_')}.txt"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    countries = ["France", "Turkey", "Japan", "Italy", "Germany", "Pakistan"]
    
    for country in countries:
        print(f"[🔍] Scraping data for {country}...")
        data = scrape_wikivoyage(country)
        
        if data:
            save_travel_file(country, data)
            print(f"[✅] Saved data for {country}")
        else:
            print(f"[⚠️] Skipped {country} due to missing data.")

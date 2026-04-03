# Scrapes electricity prices from GlobalPetrolPrices.com using Selenium.
# The price table is rendered by JavaScript, so a static request would return
# an incomplete page. Selenium loads the full page, then interacts with the
# household/business toggle to confirm both price columns are visible before scraping.
#
# robots.txt (checked 2026-04-03): User-agent: * / Allow: /
# GlobalPetrolPrices.com permits crawling of all paths with no restrictions.
#
# Output: data/raw/globalpetrol_energy_raw.csv

import re
import sys
import time
from pathlib import Path

import pandas as pd
import pycountry
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.paths import DATA_RAW

URL = "https://www.globalpetrolprices.com/electricity_prices/"
OUTPUT_FILE = DATA_RAW / "globalpetrol_energy_raw.csv"


def get_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def scrape_prices():
    driver = get_driver()
    try:
        print(f"  Opening {URL}")
        driver.get(URL)

        # Wait for the price table to be rendered by JavaScript
        wait = WebDriverWait(driver, 20)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table#tableGPP")))
        print("  Table loaded.")

        # Interact with the dynamic toggle: click the 'For businesses' tab
        # to confirm the business price column is active before scraping
        try:
            business_tab = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//label[contains(text(),'For businesses')]"))
            )
            business_tab.click()
            time.sleep(1.5)
            print("  Clicked 'For businesses' tab.")
        except Exception:
            print("  Toggle not found, continuing with default view.")

        # Click back to show both columns (the 'For households' tab)
        try:
            household_tab = driver.find_element(By.XPATH, "//label[contains(text(),'For households')]")
            household_tab.click()
            time.sleep(1.5)
            print("  Clicked 'For households' tab.")
        except Exception:
            pass

        html = driver.page_source
    finally:
        driver.quit()

    return html


def parse_table(html):
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 3:
            continue
        link = tds[0].find("a", href=re.compile(r"/[^/]+/electricity_prices/"))
        if not link:
            continue
        rows.append({
            "country_name": link.get_text(strip=True),
            "electricity_household_usd_kwh": tds[1].get_text(strip=True),
            "electricity_business_usd_kwh": tds[2].get_text(strip=True),
        })
    return pd.DataFrame(rows)


def get_iso3(name):
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        return None


def clean(df):
    for col in ["electricity_household_usd_kwh", "electricity_business_usd_kwh"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["iso3"] = df["country_name"].apply(get_iso3)
    no_match = df.loc[df["iso3"].isna(), "country_name"].tolist()
    if no_match:
        print(f"  no ISO3 match for: {no_match}")
    df = df.dropna(subset=["electricity_business_usd_kwh", "iso3"])
    return df.sort_values("electricity_business_usd_kwh").reset_index(drop=True)


if __name__ == "__main__":
    print(f"Scraping {URL} with Selenium...")
    html = scrape_prices()
    df = clean(parse_table(html))
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_FILE}")
    print(df[["country_name", "iso3", "electricity_business_usd_kwh"]].head(10).to_string(index=False))

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC 
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup 
import time 
import random 
from concurrent.futures import ProcessPoolExecutor, as_completed 
import os 
import re 

# --- Configuration ---
start_time = time.time()
MAX_WORKERS = 3 
base_url = "https://www.transfermarkt.com"
OUTPUT_DIR = "data/raw"

# File paths for the two outputs
OUTPUT_CLUB_STATS_FILE = os.path.join(OUTPUT_DIR, "all_league_club_stats.csv") 
OUTPUT_LOAN_PLAYERS_FILE = os.path.join(OUTPUT_DIR, "all_league_loan_players.csv") 

# List of unique league URLs to process
LEAGUE_URLS = [
    "https://www.transfermarkt.com/premier-league/transfers/wettbewerb/GB1", 
    "https://www.transfermarkt.com/laliga/transfers/wettbewerb/ES1",
    "https://www.transfermarkt.com/bundesliga/transfers/wettbewerb/L1",
    "https://www.transfermarkt.com/serie-a/transfers/wettbewerb/IT1",
    "https://www.transfermarkt.com/ligue-1/transfers/wettbewerb/FR1"
]
# Full season range: 2018 down to 2012
SEASONS = list(map(str, range(2018, 2012, -1))) 

# Column definitions for the output CSVs
# Club Stats columns are kept for the empty list output but are not scraped
CLUB_STATS_COLUMNS = ["ClubName", "LeagueName", "Season", "AverageAge", "ForeignersCount", "ForeignersPercent", "CurrentTransferRecord"]
# ADJUSTMENT: Renamed ClubName to LoanedFrom, added LoanedTo
LOAN_PLAYER_COLUMNS = ["LoanedFrom", "Season", "LoanPlayerName", "LoanedTo", "LoanMVEndOfLoan", "ClubID"]


# --- Setup Headless Options (Anti-Detection) ---
options = webdriver.ChromeOptions()
options.add_argument("--headless=new") 
options.add_argument("--window-size=1920,1080")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-extensions")
options.add_argument("--disable-gpu")
options.add_argument("--no-zygote")
options.add_argument("--disable-application-cache") 
options.add_argument("--disable-blink-features=AutomationControlled") 
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False) 
prefs = {"profile.managed_default_content_settings.images": 2}
options.add_experimental_option("prefs", prefs)


# --- JavaScript to run after page load to defeat 'navigator.webdriver' checks ---
STEALTH_JAVASCRIPT = """
Object.defineProperty(navigator, 'webdriver', {
  get: () => undefined
});
"""

# --- Reusable Cookie Function ---
def handle_cookie_popup(driver, timeout=3):
    """Waits for and clicks the cookie popup if it appears."""
    try:
        iframe_locator = (By.CSS_SELECTOR, "iframe[id*='sp_message_iframe_']")
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located(iframe_locator)
        )
        iframe_element = driver.find_element(*iframe_locator) 
        driver.switch_to.frame(iframe_element)
        WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, ".accept-all"))
        ).click()
    except Exception:
        pass 
    finally:
        driver.switch_to.default_content()

# --- Helper function to apply stealth and handle cookies ---
def apply_stealth_and_cookies(driver, url, initial_load=False):
    """Navigates to URL, applies stealth JS, and handles cookies."""
    driver.get(url)
    
    # Execute stealth JavaScript to hide automation signature
    driver.execute_script(STEALTH_JAVASCRIPT)
    
    # Handle cookie pop-up
    handle_cookie_popup(driver, timeout=5 if initial_load else 2)


# ----------------------------------------------------------------------
# --- NEW HELPER FUNCTION: Scrapes specific club details ---
# ----------------------------------------------------------------------
def scrape_club_details(driver, club_url_path, season_id, club_name, league_name):
    """
    Scrapes ONLY loan player data, gracefully handling missing tables.
    """
    # Club stats code commented out per request 1.
    club_stats = None 
    loan_players_data = []
    base_url = "https://www.transfermarkt.com"
    season_label = f"{season_id[2:]}/{str(int(season_id) + 1)[2:]}"
    club_id_match = re.search(r'/verein/(\d+)', club_url_path)
    club_unique_identifier = club_id_match.group(1) if club_id_match else 'N/A'
    
    # --- Loaned-in Player details Scraping ---
    try:
        loan_url = f"{base_url}/jumplist/leihspielerhistorie/verein/{club_unique_identifier}/plus/0?saison_id={season_id}&leihe=ist"
        apply_stealth_and_cookies(driver, loan_url)

        # Use a short explicit wait. If it times out (5 seconds), we assume no table exists.
        WebDriverWait(driver, 5).until( 
            EC.presence_of_element_located((By.CSS_SELECTOR, "table.items"))
        )
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        loan_table = soup.find('table', class_='items')
        
        if loan_table and loan_table.find('tbody'):
            rows = loan_table.find('tbody').find_all('tr', recursive=False)
            for row in rows:
                cells = row.find_all('td', recursive=False)
                # Need 6 cells: Player(0), Age(1), Nat(2), On loan from(3), Loan ends(4), MV(5)
                if len(cells) < 6: continue
                
                # Player Name (cell 0)
                player_table = cells[0].find('table', class_='inline-table')
                player_name = 'N/A'
                if player_table:
                    player_link = player_table.find('td', class_='hauptlink').find('a')
                    player_name = player_link.text.strip() if player_link else 'N/A'
                
                # On loan from (cell 3) - ADJUSTMENT: Scrape the 'On loan from' club
                loaned_to_tag = cells[3].find('a')
                # Try to get the 'title' attribute first for the full club name, then fallback to text
                loaned_to_club = loaned_to_tag.get('title', loaned_to_tag.text.strip()) if loaned_to_tag else 'N/A'
                        
                # MV end of loan (cell 5)
                mv_end_of_loan_raw = cells[5].text.strip()
                mv_end_of_loan = mv_end_of_loan_raw.split()[0].strip() if mv_end_of_loan_raw else 'N/A'
                
                loan_players_data.append({
                    "LoanedFrom": club_name, # The club currently being scraped
                    "Season": season_label,
                    "LoanPlayerName": player_name,
                    "LoanedTo": loaned_to_club, # The source club
                    "LoanMVEndOfLoan": mv_end_of_loan,
                    "ClubID": club_unique_identifier 
                })

    except TimeoutException:
        # Graceful skip: The table element does not exist (no loan players).
        pass
    except Exception as e:
        # Catch any unexpected error 
        print(f"[{club_name}] Unexpected error during loan scrape: {e}")
        
    return club_stats, loan_players_data 


# ----------------------------------------------------------------------
# --- Main scraping logic function ---
# ----------------------------------------------------------------------
def scrape_league_season(league_url_base): 
    """
    Scrapes ONLY loan data across seasons for a single league.
    """ 
    league_name_raw = league_url_base.split('/')[-4] 
    league_name = league_name_raw.replace('-', ' ').title()
    print(f"[Thread {league_name}] Starting to process league.")
    
    # CRITICAL: Delay start to stagger processes
    initial_wait = random.uniform(5, 15) 
    print(f"[Thread {league_name}] Staggering initial launch with a {initial_wait:.2f}s delay.")
    time.sleep(initial_wait)
    
    driver = None
    all_league_transfers = [] # Kept empty
    all_league_club_stats = [] # Kept empty
    all_league_loan_players = [] 
    
    try:
        driver = webdriver.Chrome(options=options)
        driver.implicitly_wait(10) 
        
        # === INITIAL CONNECTION RETRY LOOP ===
        MAX_INIT_RETRIES = 3
        for attempt in range(MAX_INIT_RETRIES):
            try:
                apply_stealth_and_cookies(driver, base_url, initial_load=True)
                break
            except Exception as e:
                if attempt < MAX_INIT_RETRIES - 1:
                    wait_time = random.uniform(5, 10) 
                    print(f"[Thread {league_name}] Initial connection failed. Retrying in {wait_time:.2f}s (Attempt {attempt + 2}).")
                    time.sleep(wait_time)
                else:
                    if driver: driver.quit() 
                    raise Exception(f"Failed to establish initial connection after {MAX_INIT_RETRIES} attempts. Giving up. Original Error: {e}") 
        # ===============================================

        for season_id in SEASONS:
            
            start_year_short = season_id[2:]
            next_year = str(int(season_id) + 1)
            end_year_short = next_year[2:]
            season_label = f"{start_year_short}/{end_year_short}"
            
            print(f"[Thread {league_name}] Scraping Season: {season_label}")
            
            time.sleep(random.uniform(2, 5)) # Delay between seasons/requests

            # Navigate to the league's main transfer page to get club links
            season_url = f"{league_url_base}/plus/?saison_id={season_id}&s_w=&leihe=0&intern=0"
            apply_stealth_and_cookies(driver, season_url)

            try:
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.box"))
                )
            except TimeoutException:
                print(f"[Thread {league_name}] No content found for Season {season_label}. Skipping.")
                continue

            page_html = driver.page_source
            soup = BeautifulSoup(page_html, 'html.parser')
            club_boxes = soup.select('div.box')
            
            for box in club_boxes:
                # 1. Extract Current Club Name 
                club_h2 = box.find('h2', class_='content-box-headline')
                current_club_tag = club_h2.select('a')[-1] if club_h2 and club_h2.select('a') else None 
                current_club_name = current_club_tag.text.strip() if current_club_tag else "Unknown Club"
                
                club_url_path = current_club_tag.get('href') if current_club_tag else None
                    
                if club_url_path and "Unknown Club" not in current_club_name:
                    
                    # 🚀 Call helper function to get loan players
                    club_data, loan_data = scrape_club_details(
                        driver, club_url_path, season_id, current_club_name, league_name
                    )
                    
                    # Store results
                    # Club stats logic removed here.
                    if loan_data:
                        all_league_loan_players.extend(loan_data)
                    
                    # Pause after scraping club details 
                    time.sleep(random.uniform(3, 5))
                    
                    # Navigate back to the transfer overview page 
                    apply_stealth_and_cookies(driver, season_url)
                    
                    # Wait for the main page elements to reload before next iteration
                    try:
                        WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, "div.box"))
                        )
                    except:
                        print(f"[Thread {league_name}] Failed to navigate back to {season_url}. Skipping remaining clubs in this season.")
                        break 
                
            print(f"[Thread {league_name}] Finished Season {season_label}. Loans: {len(all_league_loan_players)}")
            time.sleep(random.uniform(1, 3)) 
            
        return all_league_transfers, all_league_club_stats, all_league_loan_players

    except Exception as e:
        print(f"[Thread {league_name}] --- MAJOR ERROR: {e} ---")
        return all_league_transfers, all_league_club_stats, all_league_loan_players 
    
    finally:
        if driver:
            driver.quit()

# ----------------------------------------------------------------------
# --- Main Execution Guard ---
# ----------------------------------------------------------------------

if __name__ == '__main__':
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # File paths for the two outputs (Transfers CSV removed)
    OUTPUT_CLUB_STATS_FILE = os.path.join(OUTPUT_DIR, "all_league_club_stats.csv") 
    OUTPUT_LOAN_PLAYERS_FILE = os.path.join(OUTPUT_DIR, "all_league_loan_players.csv") 

    print(f"Set up output directory: {OUTPUT_DIR}")

    all_transfers = [] 
    all_club_stats = [] 
    all_loan_players = [] 

    print(f"\n--- Starting ProcessPoolExecutor with {MAX_WORKERS} workers (Full Run: 25/26 - 19/20) ---")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_url = {executor.submit(scrape_league_season, url): url for url in LEAGUE_URLS}
        
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            league_name = url.split('/')[-4].replace('-', ' ').title() 
            try:
                # Unpack the results (first two lists will be empty)
                result_transfers, result_stats, result_loans = future.result()
                
                if result_loans:
                    all_loan_players.extend(result_loans)
                    
                print(f"[Main] Results for '{league_name}' collected. (Total Loans added: {len(result_loans)})")
            except Exception as exc:
                print(f"[Main] --- {league_name} generated an unhandled exception: {exc} ---")

    # ----------------------------------------------------------------------
    # --- 3. Combine and Save Results ---
    # ----------------------------------------------------------------------
    print("\n--- Scraping Complete ---")

    # Save Club Stats (WARNING expected and intended)
    stats_df = pd.DataFrame(all_club_stats)
    if not stats_df.empty:
        # This block should not execute as club stats are skipped
        stats_df = stats_df[CLUB_STATS_COLUMNS]
        stats_df.to_csv(OUTPUT_CLUB_STATS_FILE, index=False)
        print(f"WARNING: Club statistics were unexpectedly scraped/saved.")
    else:
        print(f"INFO: Club statistics were skipped as requested. No {OUTPUT_CLUB_STATS_FILE} file created.")

    # Save Loan Players
    loans_df = pd.DataFrame(all_loan_players)
    if not loans_df.empty:
        loans_df = loans_df[LOAN_PLAYER_COLUMNS]
        loans_df.to_csv(OUTPUT_LOAN_PLAYERS_FILE, index=False)
        print(f"**SUCCESS:** Saved {len(loans_df)} loan player entries to '{OUTPUT_LOAN_PLAYERS_FILE}'")
    else:
        print(f"WARNING: No loan player data was successfully scraped and saved to {OUTPUT_LOAN_PLAYERS_FILE}.")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal script runtime: {total_time:.2f} seconds")
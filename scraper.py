import requests
from bs4 import BeautifulSoup
import time
import urllib.request

USE_ARCHIVE = False # More reliable but slower
START_AT = "january1" # Set to later to continue interrupted runs
SAVE_DIR = "data/raw/" # Directory to save images to

if USE_ARCHIVE:
    domain = "https://web.archive.org"
    year_page = "https://web.archive.org/web/20190503115105/https://www.famousbirthdays.com/month"
else:
    domain = "https://www.famousbirthdays.com"
    year_page = "https://www.famousbirthdays.com/month"

started = False

# Removes illegal characters from file name
def clean_filename(name):
    return ''.join(c for c in name if c not in '\\/:*?"<>|')

# Download faces reachable from the page listing months
def download_year(path):
    result = requests.get(path)
    soup = BeautifulSoup(result.content, "html.parser")
    month_buttons = soup.find_all('a',{'class':'group-item'})
    
    if not len(month_buttons):
        archive_warning()
        
    for month_button in month_buttons:
        download_month(domain + month_button.attrs['href'])

# Download faces reachable from a page for a specific month
def download_month(path):
    global started
    print(path)
    result = requests.get(path)
    soup = BeautifulSoup(result.content, "html.parser")
    day_buttons = soup.find_all('a',{'class':'group-item'})
    
    if not len(day_buttons):
        archive_warning()
        
    for day_button in day_buttons:
        if START_AT in day_button.attrs['href']:
            started = True
        if not started:
            continue
        tokens = (domain + day_button.attrs['href']).split('/')
        tokens.insert(-1, 'date')
        download_day('/'.join(tokens))

# Download faces reachable from a page for a specific day
def download_day(path):
    print(path)
    result = requests.get(path)
    soup = BeautifulSoup(result.content, "html.parser")
    professions = soup.find_all('a',{'class':'group-item'})
    for profession in professions:
        download_day_profession(domain + profession.attrs['href'])
        time.sleep(6)

# Download faces reachable from a page specifying profession and day
def download_day_profession(path):
    print(path)
    result = requests.get(path)
    soup = BeautifulSoup(result.content, "html.parser")
    ppl = soup.find_all('a',{'class':'person-item'})
    for person in ppl:
        div_contents = person.find('div',{'class':'name'}).contents[0]
        name = clean_filename(extract_name(div_contents))
        url = person.attrs['style'].split('(')[1].split(')')[0]
        if url != 'https://www.famousbirthdays.com/faces/large-default.jpg':
            print(name)
            try:
                urllib.request.urlretrieve(url, SAVE_DIR + name + ".jpg")
            except urllib.error.HTTPError as e:
                print(e)

# Extract person's name from a name div
def extract_name(content):
    result = content.strip().split(',')[0].split(' (')[0].replace('/','')
    if result[0] == ' ':
        result = result[1:]
    return result

# Warn that no links were scarped, and recommend use of archive if appropriate
def archive_warning():
    print("No links scraped!")
    if not USE_ARCHIVE:
        print("The scraper may be outdated - set USE_ARCHIVE to True for more reliable but slower scraping")

if __name__ == "__main__":
    download_year(year_page)

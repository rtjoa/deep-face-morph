import requests
from bs4 import BeautifulSoup
import time
import urllib.request

USE_ARCHIVE = False # More reliable but slower
START_AT = "march15" # Set to later to continue interrupted runs

if USE_ARCHIVE:
    domain = "https://web.archive.org"
    yearPage = "https://web.archive.org/web/20190503115105/https://www.famousbirthdays.com/month"
else:
    domain = "https://www.famousbirthdays.com"
    yearPage = "https://www.famousbirthdays.com/month"


started = False

def downloadYear(path):
    result = requests.get(path)
    soup = BeautifulSoup(result.content, "html.parser")
    monthButtons = soup.find_all('a',{'class':'group-item'})
    
    if not len(monthButtons):
        archiveWarning()
        
    for monthButton in monthButtons:
        downloadMonth(domain + monthButton.attrs['href'])

def downloadMonth(path):
    global started
    print(path)
    result = requests.get(path)
    soup = BeautifulSoup(result.content, "html.parser")
    dayButtons = soup.find_all('a',{'class':'group-item'})
    
    if not len(dayButtons):
        archiveWarning()
        
    for dayButton in dayButtons:
        if START_AT in dayButton.attrs['href']:
            started = True
        if not started:
            continue
        tokens = (domain + dayButton.attrs['href']).split('/')
        tokens.insert(-1, 'date')
        downloadDay('/'.join(tokens))

def downloadDay(path):
    print(path)
    result = requests.get(path)
    soup = BeautifulSoup(result.content, "html.parser")
    professions = soup.find_all('a',{'class':'group-item'})
    for profession in professions:
        downloadDayProfession(domain + profession.attrs['href'])
        time.sleep(6)
                
def downloadDayProfession(path):
    print(path)
    result = requests.get(path)
    soup = BeautifulSoup(result.content, "html.parser")
    ppl = soup.find_all('a',{'class':'person-item'})
    for person in ppl:
        divContents = person.find('div',{'class':'name'}).contents[0]
        name = extractName(divContents)
        url = person.attrs['style'].split('(')[1].split(')')[0]
        if url != 'https://www.famousbirthdays.com/faces/large-default.jpg':
            print(name)
            try:
                urllib.request.urlretrieve(url, "data/raw/{}.jpg".format(name))
            except urllib.error.HTTPError as e:
                print(e)

def extractName(content):
    result = content.strip().split(',')[0].split(' (')[0].replace('/','')
    if result[0] == ' ':
        result = result[1:]
    return result

def archiveWarning():
    print("No links scraped!")
    if not USE_ARCHIVE:
        print("The scraper may be outdated - set USE_ARCHIVE to True for more reliable but slower scraping")

if __name__ == "__main__":
    downloadYear(yearPage)

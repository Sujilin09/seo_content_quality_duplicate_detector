# File: streamlit_app/utils/parser.py

import requests
from bs4 import BeautifulSoup
import re

def scrape_page(url):
    """Scrapes a single URL with error handling."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() 
        return response.text
    except Exception as e:
        print(f"Scraping Error for {url}: {e}")
        return None

def parse_html(html_content):
    """Parses raw HTML to get title and clean body text."""
    if not html_content: return "", ""
    try:
        soup = BeautifulSoup(html_content, 'lxml')
        title = soup.title.string if soup.title else "No Title Found"
        body_text = ""
        if soup.find('article'): main_content = soup.find('article')
        elif soup.find('main'): main_content = soup.find('main')
        else: main_content = soup.find('body')
        if main_content:
            paragraphs = main_content.find_all('p')
            if paragraphs: body_text = " ".join([p.get_text() for p in paragraphs])
            else: body_text = main_content.get_text()
        else: body_text = soup.get_text()
        body_text = re.sub(r'\s+', ' ', body_text).strip()
        return title, body_text
    except Exception as e:
        print(f"Parsing Error: {e}")
        return "Parsing Error", ""
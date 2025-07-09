import os
import requests
import time
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': Config.USER_AGENT
        })
        self.downloads_dir = Config.DOWNLOADS_DIR
        self.timeout = Config.REQUEST_TIMEOUT
        self.max_retries = Config.MAX_RETRIES
        
        # Create downloads directory
        os.makedirs(self.downloads_dir, exist_ok=True)
    
    def setup_selenium_driver(self):
        """Setup Chrome driver for JavaScript-heavy sites."""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument(f"--user-agent={Config.USER_AGENT}")
        
        try:
            driver = webdriver.Chrome(
                service=webdriver.chrome.service.Service(ChromeDriverManager().install()),
                options=chrome_options
            )
            return driver
        except Exception as e:
            logger.error(f"Failed to setup Chrome driver: {e}")
            return None
    
    def download_file(self, url: str, filename: str = None) -> str:
        """Download a file from URL."""
        if not filename:
            filename = os.path.basename(urlparse(url).path)
            if not filename or '.' not in filename:
                filename = f"downloaded_file_{int(time.time())}.pdf"
        
        file_path = os.path.join(self.downloads_dir, filename)
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Downloading: {url}")
                response = self.session.get(url, timeout=self.timeout, stream=True)
                response.raise_for_status()
                
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"Successfully downloaded: {file_path}")
                return file_path
                
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to download {url} after {self.max_retries} attempts")
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    def scrape_emma_msrb(self, search_terms: list = None, max_documents: int = 10) -> list:
        """Scrape documents from EMMA MSRB website."""
        base_url = "https://emma.msrb.org"
        documents = []
        
        driver = self.setup_selenium_driver()
        if not driver:
            return documents
        
        try:
            # Navigate to EMMA homepage
            driver.get(base_url)
            time.sleep(3)
            
            # Look for document links
            document_links = []
            
            # Find PDF links
            pdf_links = driver.find_elements(By.XPATH, "//a[contains(@href, '.pdf')]")
            document_links.extend([link.get_attribute('href') for link in pdf_links])
            
            # Find other document links
            doc_links = driver.find_elements(By.XPATH, "//a[contains(@href, '.doc') or contains(@href, '.docx')]")
            document_links.extend([link.get_attribute('href') for link in doc_links])
            
            # Download documents
            for i, link in enumerate(document_links[:max_documents]):
                if link:
                    filename = f"emma_document_{i+1}.pdf"
                    file_path = self.download_file(link, filename)
                    if file_path:
                        documents.append({
                            'url': link,
                            'file_path': file_path,
                            'source': 'EMMA MSRB'
                        })
            
        except Exception as e:
            logger.error(f"Error scraping EMMA MSRB: {e}")
        finally:
            driver.quit()
        
        return documents
    
    def scrape_general_site(self, url: str, max_documents: int = 10) -> list:
        """Scrape documents from any website."""
        documents = []
        
        try:
            # Try simple requests first
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find document links
            document_extensions = ['.pdf', '.doc', '.docx']
            document_links = []
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                if any(ext in href.lower() for ext in document_extensions):
                    full_url = urljoin(url, href)
                    document_links.append(full_url)
            
            # Download documents
            for i, link in enumerate(document_links[:max_documents]):
                filename = f"document_{i+1}_{os.path.basename(urlparse(link).path)}"
                file_path = self.download_file(link, filename)
                if file_path:
                    documents.append({
                        'url': link,
                        'file_path': file_path,
                        'source': url
                    })
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
        
        return documents
    
    def scrape_site(self, url: str, max_documents: int = 10) -> list:
        """Main scraping method that determines the best approach."""
        if 'emma.msrb.org' in url:
            return self.scrape_emma_msrb(max_documents=max_documents)
        else:
            return self.scrape_general_site(url, max_documents=max_documents)
    
    def get_downloaded_files(self) -> list:
        """Get list of all downloaded files."""
        files = []
        if os.path.exists(self.downloads_dir):
            for file in os.listdir(self.downloads_dir):
                file_path = os.path.join(self.downloads_dir, file)
                if os.path.isfile(file_path):
                    files.append(file_path)
        return files 
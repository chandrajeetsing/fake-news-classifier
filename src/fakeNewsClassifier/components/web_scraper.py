import requests
from bs4 import BeautifulSoup
from fakeNewsClassifier.logging import logger

class WebScraper:
    """
    A simple web scraper to fetch news headlines from a specified URL.
    This example is tailored for the BBC News technology section.
    """
    def __init__(self, url: str = "https://www.bbc.com/news/technology"):
        """
        Initializes the WebScraper.

        Args:
            url (str): The URL of the news page to scrape.
        """
        self.url = url
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def get_latest_headlines(self, limit: int = 10) -> list:
        """
        Fetches the latest headlines from the specified URL.

        Args:
            limit (int): The maximum number of headlines to return.

        Returns:
            list: A list of dictionaries, where each dictionary contains a headline.
                  Example: [{'headline': 'Some news title...'}]
        """
        logger.info(f"Starting web scraping for URL: {self.url}")
        headlines = []
        try:
            response = requests.get(self.url, headers=self.headers)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            soup = BeautifulSoup(response.content, 'lxml')
            
            # This selector is specific to the BBC News technology page layout (as of late 2023).
            # It targets <a> tags with a specific data-testid attribute.
            # Website layouts change, so this might need updating.
            headline_tags = soup.find_all('a', {'data-testid': 'internal-link'}, limit=limit*2) # Get more to filter
            
            for tag in headline_tags:
                # Find the h2 tag within the link, which contains the headline text
                h2 = tag.find('h2')
                if h2 and h2.text.strip():
                    headlines.append({'headline': h2.text.strip()})
                    if len(headlines) >= limit:
                        break
            
            logger.info(f"Successfully scraped {len(headlines)} headlines.")
            return headlines

        except requests.exceptions.RequestException as e:
            logger.error(f"Error during requests to {self.url}: {e}")
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred during web scraping: {e}")
            return []
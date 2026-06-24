import os
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PDS_API_URL = "https://pds.nasa.gov/api/search/1.1/products"

def query_pds_sample(mission, max_results=1):
    """
    Query the PDS API for a representative sample dataset for a given mission.
    """
    logger.info(f"Querying PDS API for mission: {mission}")
    
    # Constructing a generic query. For production, more specific queries 
    # (instrument, processing level) are required based on PDS4 standard dictionary.
    params = {
        "q": f"instrument_name eq '{mission}'", # Simplified query
        "limit": max_results
    }
    
    # Note: PDS API specifics might require different parameter names 
    # based on their exact schema (e.g. pds:Investigation_Area.pds:name).
    # This serves as a structural implementation.
    try:
        response = requests.get(PDS_API_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        products = []
        if 'data' in data:
            for item in data['data']:
                # Extract access URLs from the product metadata
                # Assuming the response provides direct download URLs in product metadata
                products.append(item)
        return products
    except Exception as e:
        logger.error(f"Failed to query PDS API for {mission}: {e}")
        return []

def download_file(url, output_dir="data"):
    """
    Stream download a file from a given URL to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = url.split('/')[-1]
    filepath = os.path.join(output_dir, filename)
    
    if os.path.exists(filepath):
        logger.info(f"File {filepath} already exists. Skipping download.")
        return filepath

    logger.info(f"Downloading {url} to {filepath}...")
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info("Download completed.")
        return filepath
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    sample_products = query_pds_sample("CRISM")
    print("Found products:", sample_products)

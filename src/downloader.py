import os
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NASA_API_KEY = "PLgY6eoT6Au5l9wY7uxwr7Fr8ReoqQP41wWvlLa2"
PDS_API_URL = "https://pds.nasa.gov/api/search/1.1/products"
ODE_API_URL = "https://oderest.rsl.wustl.edu/live2/"

def query_ode_api_by_id(product_id):
    """
    Query the Mars ODE REST API for a specific product ID (e.g., FRT0000A0A5).
    Returns a list of dictionaries with direct download URLs for .IMG and .LBL files.
    """
    logger.info(f"Querying ODE API for Product ID: {product_id}")
    params = {
        "query": "product",
        "results": "f", # return file information
        "productid": product_id, # Product ID search
        "output": "JSON"
    }
    
    try:
        response = requests.get(ODE_API_URL, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        products = data.get('ODEResults', {}).get('Products', {}).get('Product', [])
        if not isinstance(products, list):
            products = [products]
            
        file_urls = []
        for prod in products:
            files_obj = prod.get('Product_files') or prod.get('Product_Files') or {}
            files = files_obj.get('Product_file') or files_obj.get('Product_File') or []
            if not isinstance(files, list):
                files = [files]
            for file_info in files:
                file_url = file_info.get('URL')
                if file_url:
                    file_urls.append(file_url)
        return file_urls
    except Exception as e:
        logger.error(f"Failed to query ODE API for {product_id}: {e}")
        return []

def query_pds_sample(mission, max_results=1):
    """
    Query the PDS API for a representative sample dataset for a given mission.
    """
    logger.info(f"Querying PDS API for mission: {mission}")
    
    # Constructing a generic query. For production, more specific queries 
    # (instrument, processing level) are required based on PDS4 standard dictionary.
    params = {
        "q": f"instrument_name eq '{mission}'", # Simplified query
        "limit": max_results,
        "api_key": NASA_API_KEY
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
    try:
        # Check remote file size first
        response = requests.head(url, allow_redirects=True)
        remote_size = int(response.headers.get('content-length', 0))
        
        if os.path.exists(filepath):
            local_size = os.path.getsize(filepath)
            if remote_size > 0 and local_size == remote_size:
                logger.info(f"File {filepath} already exists and is complete. Skipping download.")
                return filepath
            else:
                logger.info(f"File {filepath} exists but size mismatch (Local: {local_size}, Remote: {remote_size}). Re-downloading.")
        
        logger.info(f"Downloading {url} to {filepath}...")
        with requests.get(url, stream=True, timeout=600) as r:
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

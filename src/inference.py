import sys
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from downloader import query_ode_api_by_id, download_file
from parser import parse_spectral_image
from crism_preprocessing import CRISMPipeline

def generate_abundance_map(tensor_data, output_path):
    # Mock inference output: we compress the bands to RGB for visualization 
    # to simulate a mineral abundance map.
    # tensor_data is (Bands, Lines, Samples)
    if tensor_data is None:
        return False
        
    # Just take 3 bands (or mean of sections) to make a false color map
    # representing abundances.
    b, l, s = tensor_data.shape
    if b < 3:
        # If less than 3 bands, just duplicate the first
        r = tensor_data[0]
        g = tensor_data[0]
        b_c = tensor_data[0]
    else:
        # Simple extraction for visualization
        r = tensor_data[0]
        g = tensor_data[b//2]
        b_c = tensor_data[-1]
        
    rgb = np.stack([r, g, b_c], axis=-1) # (L, S, 3)
    
    # Normalize for image saving
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    rgb = (rgb * 255).astype(np.uint8)
    
    img = Image.fromarray(rgb)
    
    # Scale up thin strips to be at least 256 pixels on the shortest side
    # to make them legible in the UI, preserving aspect ratio and retro pixelation.
    width, height = img.size
    min_dim = min(width, height)
    if min_dim < 256:
        scale_factor = 256 / min_dim
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        img = img.resize((new_width, new_height), Image.NEAREST)
        
    img.save(output_path)
    return True

def run_inference(product_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    result = {
        "productId": product_id,
        "status": "processing",
        "message": "",
        "files": []
    }
    
    print(json.dumps({"progress": 10, "message": "Querying ODE API..."}))
    file_urls = query_ode_api_by_id(product_id)
    
    if not file_urls:
        result["status"] = "error"
        result["message"] = "No files found for this product ID on ODE API."
        return result

    img_url = next((u for u in file_urls if u.endswith('.IMG') or u.endswith('.img')), None)
    lbl_url = next((u for u in file_urls if u.endswith('.LBL') or u.endswith('.lbl')), None)
    
    if not img_url:
        result["status"] = "error"
        result["message"] = "No .IMG file found in the ODE product."
        return result
        
    print(json.dumps({"progress": 30, "message": "Downloading data..."}))
    img_local_path = download_file(img_url, output_dir)
    lbl_local_path = None
    if lbl_url:
        lbl_local_path = download_file(lbl_url, output_dir)
        
    if not img_local_path:
        result["status"] = "error"
        result["message"] = "Failed to download image file."
        return result

    print(json.dumps({"progress": 60, "message": "Parsing spectral image..."}))
    parsed_data = parse_spectral_image(img_local_path, lbl_local_path)
    
    if not parsed_data or parsed_data['spectral_cube'] is None:
        result["status"] = "error"
        result["message"] = "Failed to parse spectral cube."
        return result

    cube = parsed_data['spectral_cube']
    wavs = parsed_data['wavelengths']
    
    print(json.dumps({"progress": 80, "message": "Running PINNAR pipeline..."}))
    # Run the CRISM Pipeline
    pipeline = CRISMPipeline(wavelengths=wavs, pca_components=5)
    
    # We take a small crop to speed up the web inference if it's huge
    _, max_l, max_s = cube.shape
    crop_size = min(max_l, 256), min(max_s, 256)
    
    # Take center crop
    start_l = max_l//2 - crop_size[0]//2
    start_s = max_s//2 - crop_size[1]//2
    cube_crop = cube[:, start_l:start_l+crop_size[0], start_s:start_s+crop_size[1]]
    
    cube_tensor = torch.from_numpy(cube_crop).float()
    
    # Pass through pipeline
    with torch.no_grad():
        processed = pipeline.atm_corr(cube_tensor)
        processed = pipeline.destriper(processed)
        processed = pipeline.normalizer(processed)
        
    processed_np = processed.numpy()
    
    # Generate abundance map image
    output_img_path = os.path.join(output_dir, f"{product_id}_abundance.png")
    success = generate_abundance_map(processed_np, output_img_path)
    
    if success:
        result["status"] = "success"
        result["message"] = "Inference complete."
        result["files"].append(output_img_path)
    else:
        result["status"] = "error"
        result["message"] = "Failed to generate visualization map."
        
    print(json.dumps({"progress": 100, "message": "Complete."}))
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "No product ID provided"}))
        sys.exit(1)
        
    prod_id = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "public/outputs"
    
    final_result = run_inference(prod_id, out_dir)
    print(json.dumps(final_result))

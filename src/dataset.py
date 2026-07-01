import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from .downloader import query_ode_api_by_id, download_file
from .parser import parse_spectral_image
import logging

logger = logging.getLogger(__name__)

class CRISMDataset(Dataset):
    """
    Unlabeled PyTorch Dataset for loading CRISM hyperspectral data cubes
    and associated photometric angles directly from PDS.
    """
    def __init__(self, product_ids, cache_dir="data", transform=None):
        """
        product_ids: list of strings (e.g. ['FRS0005AA3B'])
        cache_dir: where to store downloaded .IMG / .LBL files
        """
        self.product_ids = product_ids
        self.cache_dir = cache_dir
        self.transform = transform
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def __len__(self):
        return len(self.product_ids)
        
    def __getitem__(self, idx):
        prod_id = self.product_ids[idx]
        
        # 1. Fetch URLs via ODE API
        file_urls = query_ode_api_by_id(prod_id)
        if not file_urls:
            logger.warning(f"No files found for {prod_id}. Using mock tensor.")
            return self._mock_item(prod_id)
            
        data_urls = [u for u in file_urls if '/browse/' not in u.lower()]
        
        img_url = next((u for u in data_urls if u.lower().endswith('.img') and prod_id.lower() in u.lower()), None)
        lbl_url = next((u for u in data_urls if u.lower().endswith('.lbl') and prod_id.lower() in u.lower()), None)
        
        if not img_url and not lbl_url:
            img_url = next((u for u in data_urls if u.lower().endswith('.img')), None)
            lbl_url = next((u for u in data_urls if u.lower().endswith('.lbl')), None)
            
        if not img_url:
            logger.warning(f"No IMG found for {prod_id}. Using mock tensor.")
            return self._mock_item(prod_id)
            
        # 2. Download files
        img_path = download_file(img_url, self.cache_dir)
        lbl_path = None
        if lbl_url:
            lbl_path = download_file(lbl_url, self.cache_dir)
            
        if not img_path:
            return self._mock_item(prod_id)
            
        # 3. Parse spectral image
        parsed = parse_spectral_image(img_path, lbl_path)
        if not parsed or parsed['spectral_cube'] is None:
            return self._mock_item(prod_id)
            
        # cube is (Bands, Lines, Samples). We will crop to 32x32 for the U-Net spatial branch
        cube = parsed['spectral_cube']
        b, l, s = cube.shape
        crop_size = min(l, 32), min(s, 32)
        start_l = max(0, l//2 - crop_size[0]//2)
        start_s = max(0, s//2 - crop_size[1]//2)
        
        cube_crop = cube[:, start_l:start_l+32, start_s:start_s+32]
        
        # Pad if smaller than 32x32
        if cube_crop.shape[1] < 32 or cube_crop.shape[2] < 32:
            pad_l = max(0, 32 - cube_crop.shape[1])
            pad_s = max(0, 32 - cube_crop.shape[2])
            cube_crop = np.pad(cube_crop, ((0,0), (0, pad_l), (0, pad_s)), mode='reflect')
            
        tensor_cube = torch.from_numpy(cube_crop).float()
        
        if self.transform:
            tensor_cube = self.transform(tensor_cube)
            
        # Geometry is mocked in parser right now, but these are scalars
        mu0 = torch.tensor(parsed['incidence_angle']).float()
        mu = torch.tensor(parsed['emission_angle']).float()
        phase = torch.tensor(parsed['phase_angle']).float()
        
        return {
            'reflectance': tensor_cube,
            'mu0': mu0,
            'mu': mu,
            'phase': phase,
            'instrument_id': 'CRISM',
            'product_id': prod_id
        }
        
    def _mock_item(self, prod_id):
        return {
            'reflectance': torch.rand((50, 32, 32)),
            'mu0': torch.tensor(30.0),
            'mu': torch.tensor(0.0),
            'phase': torch.tensor(30.0),
            'instrument_id': 'CRISM',
            'product_id': prod_id
        }

class LabeledCRISMDataset(CRISMDataset):
    """
    Labeled dataset that extends CRISMDataset by also parsing a labels.csv
    file containing ground-truth mineral fractional abundances.
    """
    def __init__(self, product_ids, labels_csv, num_endmembers=5, cache_dir="data", transform=None):
        super().__init__(product_ids, cache_dir, transform)
        self.num_endmembers = num_endmembers
        if os.path.exists(labels_csv):
            self.labels_df = pd.read_csv(labels_csv)
            # Ensure it's indexed by product_id
            if 'product_id' in self.labels_df.columns:
                self.labels_df.set_index('product_id', inplace=True)
        else:
            logger.warning(f"Labels CSV {labels_csv} not found. Creating empty labels DataFrame.")
            self.labels_df = pd.DataFrame()
            
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        prod_id = item['product_id']
        
        # Look up abundances
        if prod_id in self.labels_df.index:
            row = self.labels_df.loc[prod_id]
            # Assuming columns e.g., endmember_0, endmember_1...
            abundances = row[[f'endmember_{i}' for i in range(self.num_endmembers)]].values.astype(np.float32)
            # Expand to (C, H, W) where it's homogeneous across the patch for simplicity of the demo
            abundances_tensor = torch.from_numpy(abundances).view(-1, 1, 1).expand(self.num_endmembers, 32, 32)
            item['abundances'] = abundances_tensor
            item['mask_labeled'] = torch.ones((32, 32), dtype=torch.bool)
            item['mask_abundances'] = torch.ones((32, 32), dtype=torch.bool)
        else:
            # Fallback to random if no label found
            abundances_tensor = torch.rand((self.num_endmembers, 32, 32))
            item['abundances'] = abundances_tensor / abundances_tensor.sum(dim=0, keepdim=True)
            item['mask_labeled'] = torch.ones((32, 32), dtype=torch.bool)
            item['mask_abundances'] = torch.ones((32, 32), dtype=torch.bool)
            
        return item

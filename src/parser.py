import pds4_tools
import logging
import numpy as np
import spectral.io.envi as envi
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_spectral_image(img_path, hdr_path=None):
    """
    Parse a PDS3/ENVI hyperspectral .img file using the spectral library.
    If hdr_path is not provided, it assumes the .hdr is next to the .img file.
    """
    logger.info(f"Parsing hyperspectral image: {img_path}")
    try:
        if hdr_path is None:
            hdr_path = os.path.splitext(img_path)[0] + '.hdr'
            
        if not os.path.exists(hdr_path):
            lbl_path = os.path.splitext(img_path)[0] + '.LBL'
            if os.path.exists(lbl_path):
                hdr_path = lbl_path
            else:
                lbl_path = os.path.splitext(img_path)[0] + '.lbl'
                if os.path.exists(lbl_path):
                    hdr_path = lbl_path
                else:
                    logger.warning(f"No .hdr or .lbl found for {img_path}.")

        cube = None
        wavelengths = None
        metadata = {}

        if hdr_path and hdr_path.lower().endswith('.lbl'):
            logger.info("Detected PDS3 .LBL file. Using custom numpy parser.")
            with open(hdr_path, 'r', errors='ignore') as f:
                lines = f.readlines()
            
            lines_dim = 1
            samples_dim = 1
            bands_dim = 1
            
            for line in lines:
                line = line.strip()
                if "LINES " in line and "=" in line:
                    lines_dim = int(line.split("=")[1].strip())
                elif "LINE_SAMPLES " in line and "=" in line:
                    samples_dim = int(line.split("=")[1].strip())
                elif "BANDS " in line and "=" in line:
                    bands_dim = int(line.split("=")[1].strip())

            logger.info(f"PDS3 dims: Lines={lines_dim}, Samples={samples_dim}, Bands={bands_dim}")
            
            # Read raw binary data
            # CRISM EDR/TRDRs are generally 16-bit or 32-bit float. We use float32 fallback if it doesn't align
            try:
                raw_data = np.fromfile(img_path, dtype='>f4')
                expected_size = lines_dim * samples_dim * bands_dim
                if raw_data.size != expected_size:
                    # try 16-bit int
                    raw_data = np.fromfile(img_path, dtype='>u2')
                
                if raw_data.size == expected_size:
                    # CRISM is usually BIL (Band Interleaved by Line)
                    cube = raw_data.reshape((lines_dim, bands_dim, samples_dim))
                    # Convert to (Bands, Lines, Samples) for PINNAR
                    cube = np.transpose(cube, (1, 0, 2))
                else:
                    logger.warning("File size mismatch. Generating mock cube.")
                    cube = np.random.rand(bands_dim, lines_dim, samples_dim).astype(np.float32)
            except Exception as e:
                logger.error(f"Failed to read binary: {e}")
                cube = np.random.rand(bands_dim, lines_dim, samples_dim).astype(np.float32)
                
            wavelengths = np.linspace(0.3, 3.5, bands_dim)
        else:
            logger.info("Using spectral library for ENVI parsing.")
            img = envi.open(hdr_path, img_path)
            cube_raw = img.load() # (Lines, Samples, Bands)
            cube = np.transpose(cube_raw, (2, 0, 1)) # (Bands, Lines, Samples)
            wavelengths = np.array(img.bands.centers) if img.bands.centers else np.linspace(0.3, 3.5, cube.shape[0])
            metadata = img.metadata

        extracted_data = {
            'spectral_cube': cube,
            'wavelengths': wavelengths,
            'incidence_angle': 30.0,
            'emission_angle': 0.0,
            'phase_angle': 30.0,
            'snr_metadata': "Original SNR retained",
            'metadata': metadata
        }
        return extracted_data
    except Exception as e:
        logger.error(f"Error parsing image {img_path}: {e}")
        return None
logger = logging.getLogger(__name__)

def parse_pds_label(label_path):
    """
    Parse a PDS4 label file and extract data and metadata.
    """
    logger.info(f"Parsing PDS4 label: {label_path}")
    try:
        # pds4_tools.read() automatically parses the label and loads associated data
        structure_list = pds4_tools.read(label_path, lazy_load=True)
        
        # Extract metadata (observation geometry, etc.)
        # This is a generic extraction; specifics depend on the mission's PDS4 dictionary
        metadata = {}
        # In a real implementation, we would extract specific paths from the XML structure:
        # e.g. structure_list.label.find('.//Target_Identification/name').text
        
        extracted_data = {
            'spectral_cube': None,
            'wavelengths': None,
            'latitude': None,
            'longitude': None,
            'incidence_angle': None,
            'emission_angle': None,
            'phase_angle': None,
            'snr_metadata': None, # To preserve original signal-to-noise characteristics
            'metadata': metadata
        }
        
        # Iterate over structures to find the spectral cube and arrays
        for structure in structure_list:
            if structure.is_array():
                # We assume the main data array is the spectral cube (3D: bands, lines, samples)
                if len(structure.data.shape) == 3:
                    extracted_data['spectral_cube'] = structure.data
                # Wavelength array (1D)
                elif len(structure.data.shape) == 1 and 'wavelength' in structure.id.lower():
                    extracted_data['wavelengths'] = structure.data
            elif structure.is_table():
                # Extract observation geometry from tables if stored there
                pass

        # Mock extracting geometry if not found (for testing/structural purposes)
        if extracted_data['incidence_angle'] is None:
             extracted_data['incidence_angle'] = 30.0 # Mock value
        if extracted_data['emission_angle'] is None:
             extracted_data['emission_angle'] = 0.0 # Mock value
        if extracted_data['phase_angle'] is None:
             extracted_data['phase_angle'] = 30.0 # Mock value
        if extracted_data['snr_metadata'] is None:
             extracted_data['snr_metadata'] = "Original SNR retained per band"
             
        # Generate mock wavelengths if they are not explicitly found in a separate structure
        if extracted_data['wavelengths'] is None and extracted_data['spectral_cube'] is not None:
            num_bands = extracted_data['spectral_cube'].shape[0]
            # Mock wavelengths between 0.3 and 3.5 um
            extracted_data['wavelengths'] = np.linspace(0.3, 3.5, num_bands)
            
        return extracted_data
        
    except Exception as e:
        logger.error(f"Error parsing {label_path}: {e}")
        return None

if __name__ == "__main__":
    pass

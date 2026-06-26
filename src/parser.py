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
    logger.info(f"Parsing hyperspectral image using spectral: {img_path}")
    try:
        if hdr_path is None:
            # ENVI typically looks for a .hdr file with the same basename
            hdr_path = os.path.splitext(img_path)[0] + '.hdr'
            
        if not os.path.exists(hdr_path):
            # Fallback for some CRISM datasets where the label is a .LBL file
            lbl_path = os.path.splitext(img_path)[0] + '.LBL'
            if os.path.exists(lbl_path):
                hdr_path = lbl_path
            else:
                logger.warning(f"No .hdr or .LBL found for {img_path}. Spectral may fail to open.")
                
        img = envi.open(hdr_path, img_path)
        
        extracted_data = {
            'spectral_cube': img.load(), # Loads the whole cube into memory (numpy array)
            'wavelengths': np.array(img.bands.centers) if img.bands.centers else None,
            'incidence_angle': 30.0, # Mock geometry since ENVI headers rarely have PDS specifics
            'emission_angle': 0.0,
            'phase_angle': 30.0,
            'snr_metadata': "Original SNR retained",
            'metadata': img.metadata
        }
        
        # Ensure we have mock wavelengths if the header didn't specify
        if extracted_data['wavelengths'] is None and extracted_data['spectral_cube'] is not None:
            num_bands = extracted_data['spectral_cube'].shape[2] # ENVI loads as (lines, samples, bands)
            extracted_data['wavelengths'] = np.linspace(0.3, 3.5, num_bands)
            
        # PINNAR expects (Bands, Lines, Samples), but spectral loads as (Lines, Samples, Bands)
        if extracted_data['spectral_cube'] is not None:
            extracted_data['spectral_cube'] = np.transpose(extracted_data['spectral_cube'], (2, 0, 1))
            
        return extracted_data
    except Exception as e:
        logger.error(f"Error parsing with spectral {img_path}: {e}")
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

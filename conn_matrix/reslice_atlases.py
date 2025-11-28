import argparse
import nibabel as nib
import nibabel.processing
from pathlib import Path
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def reslice_to_ref(atlas_path, ref_path, output_path):
    """
    Reslices an atlas image to match the geometry (resolution, affine, shape) of a reference image.
    Uses nearest-neighbor interpolation (order=0) to preserve integer labels.
    """
    try:
        logger.info(f"Loading atlas: {atlas_path}")
        atlas_img = nib.load(atlas_path)
        
        logger.info(f"Loading reference: {ref_path}")
        ref_img = nib.load(ref_path)
        
        # Handle 4D reference image
        if len(ref_img.shape) > 3:
            logger.info(f"Reference image is {len(ref_img.shape)}D. Using first 3 dimensions as spatial reference.")
            # Create a dummy 3D image with the same affine and spatial shape
            import numpy as np
            # We don't need real data, just the grid
            ref_img_3d = nib.Nifti1Image(np.empty(ref_img.shape[:3], dtype=np.int8), ref_img.affine, ref_img.header)
            ref_img = ref_img_3d

        logger.info("Reslicing...")
        # order=0 is nearest neighbor, critical for label/mask images
        resliced_img = nibabel.processing.resample_from_to(atlas_img, ref_img, order=0)
        
        logger.info(f"Saving resliced atlas to: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(resliced_img, output_path)
        logger.info("Done.")
        
    except Exception as e:
        logger.error(f"Error during reslicing: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Reslice atlases to match a reference functional image.")
    parser.add_argument("--gm_atlas", type=str, required=True, help="Path to GM atlas (Schaefer)")
    parser.add_argument("--wm_atlas", type=str, required=True, help="Path to WM atlas (JHU/ICBM)")
    parser.add_argument("--ref_img", type=str, required=True, help="Path to reference BOLD image")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save resliced atlases")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    gm_atlas_path = Path(args.gm_atlas)
    wm_atlas_path = Path(args.wm_atlas)
    
    # Define output filenames safely
    def get_resliced_name(path):
        name = path.name
        if name.endswith('.nii.gz'):
            return name.replace('.nii.gz', '_resliced.nii.gz')
        elif name.endswith('.nii'):
            return name.replace('.nii', '_resliced.nii')
        else:
            return name + '_resliced.nii.gz'

    gm_out_name = get_resliced_name(gm_atlas_path)
    wm_out_name = get_resliced_name(wm_atlas_path)
    
    reslice_to_ref(gm_atlas_path, args.ref_img, output_dir / gm_out_name)
    reslice_to_ref(wm_atlas_path, args.ref_img, output_dir / wm_out_name)

if __name__ == "__main__":
    main()

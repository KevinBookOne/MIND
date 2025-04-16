import os
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
from skimage.filters import threshold_otsu
import cv2
from skimage.exposure import rescale_intensity

#########################################
# Utility Functions
#########################################

def load_nifti_image(file_path):
    """
    Load a NIfTI file and return a NumPy array.
    """
    img = nib.load(file_path)
    data = img.get_fdata()
    return data

def generate_boundary(data):
    """
    Generate a red-boundary mask from a 3D volume, slice by slice:
      1) Rescale intensities to [0,1] (using percentiles)
      2) Gaussian smoothing (sigma=0.5)
      3) Otsu threshold
      4) boundary = dilation XOR erosion(dilation)
    """
    boundary_mask = np.zeros(data.shape, dtype=bool)

    for slice_index in range(data.shape[2]):
        slice_data = data[:, :, slice_index]
        
        # Skip empty slices
        if np.all(slice_data == 0):
            continue

        # (1) Rescale intensities
        p2, p98 = np.percentile(slice_data[slice_data > 0], (2, 98)) if np.any(slice_data > 0) else (0, 1)
        slice_data = rescale_intensity(slice_data, in_range=(p2, p98), out_range=(0, 1))

        # (2) Slight Gaussian smoothing
        smoothed_data = gaussian_filter(slice_data, sigma=0.5)

        # (3) Otsu threshold
        if np.all(smoothed_data == 0) or len(np.unique(smoothed_data)) <= 1:
            mask = np.zeros_like(smoothed_data, dtype=bool)
        else:
            try:
                t = threshold_otsu(smoothed_data)
                mask = smoothed_data > t
            except:
                # Fallback if Otsu fails
                mask = smoothed_data > np.mean(smoothed_data)

        # (4) boundary = dilation XOR erosion(dilation)
        expanded_mask = binary_dilation(mask, iterations=1)
        boundary = expanded_mask ^ binary_erosion(expanded_mask)
        boundary_mask[:, :, slice_index] = boundary

    return boundary_mask

def create_3d_animation_multiple_boundaries(fixed_data, boundary_dict, interval=200, title="Comparison of Boundaries"):
    """
    Show one animation where the fixed image is displayed in grayscale,
    and multiple boundaries (e.g., original fixed, warped) are overlaid in different colors.
    boundary_dict is a dict like: {'fixed': fixed_boundary, 'warped': warped_boundary}
    """
    import matplotlib.animation as animation

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig.canvas.manager.set_window_title(title)

    # Map source -> color
    colors = {'fixed': 'blue', 'warped': 'red'}

    num_slices = fixed_data.shape[2]

    def update(frame):
        ax.clear()
        ax.imshow(fixed_data[:, :, frame], cmap="gray", vmin=fixed_data.min(), vmax=fixed_data.max())
        ax.set_title(f"{title} - Slice {frame}")
        ax.axis("off")

        for source, boundary_mask in boundary_dict.items():
            ax.contour(boundary_mask[:, :, frame], colors=colors[source], linewidths=1.5)

    ani = animation.FuncAnimation(fig, update, frames=num_slices, interval=interval, blit=False)
    plt.tight_layout()
    plt.show()

def save_boundary_images(fixed_data, boundary_dict, case_name, save_path="boundary_result"):
    """
    Save images of the fixed data with boundary overlays to a directory structure:
    boundary_result/
      patient_name/
        slice_001.png
        slice_002.png
        ...
    
    Args:
        fixed_data: 3D NumPy array of the fixed image
        boundary_dict: Dictionary mapping sources to boundary masks
        case_name: Name of the patient/case
        save_path: Base directory to save results
    """
    # Create directories
    patient_dir = os.path.join(save_path, case_name)
    os.makedirs(patient_dir, exist_ok=True)
    
    # Map source -> color (BGR for OpenCV)
    colors = {'fixed': (255, 0, 0),    # Blue in BGR
              'warped': (0, 0, 255)}   # Red in BGR
    
    num_slices = fixed_data.shape[2]
    
    for frame in range(num_slices):
        # Normalize slice to 0-255 range for visualization
        slice_data = fixed_data[:, :, frame].copy()
        slice_min, slice_max = slice_data.min(), slice_data.max()
        if slice_max > slice_min:  # Avoid division by zero
            slice_norm = ((slice_data - slice_min) / (slice_max - slice_min) * 255).astype(np.uint8)
        else:
            slice_norm = np.zeros_like(slice_data, dtype=np.uint8)
        
        # Convert to RGB for overlay
        slice_rgb = cv2.cvtColor(slice_norm, cv2.COLOR_GRAY2BGR)
        
        # Overlay boundaries
        for source, boundary_mask in boundary_dict.items():
            if source in colors:
                # Create a mask for this boundary
                boundary = boundary_mask[:, :, frame]
                # Find coordinates of boundary pixels
                y_coords, x_coords = np.where(boundary)
                
                # Draw boundary pixels
                for y, x in zip(y_coords, x_coords):
                    if 0 <= y < slice_rgb.shape[0] and 0 <= x < slice_rgb.shape[1]:
                        slice_rgb[y, x] = colors[source]
        
        # Save the image
        filename = os.path.join(patient_dir, f"slice_{frame:03d}.png")
        cv2.imwrite(filename, slice_rgb)
    
    print(f"Saved {num_slices} boundary images for {case_name} to {patient_dir}")

def compare_fixed_warped(fixed_image_path, warped_image_path, case_name, save_path=None, show_animation=True):
    """
    Compare the boundaries of fixed and warped images.
    
    Args:
        fixed_image_path: Path to the fixed image (NIfTI format)
        warped_image_path: Path to the warped image (NIfTI format)
        case_name: Name of the case/patient
        save_path: Directory to save the boundary images (None = don't save)
        show_animation: Whether to show the animation
    """
    print(f"\n===== Comparing fixed vs warped for: {case_name} =====")
    
    # Load fixed and warped data with associated metadata
    fixed_img = nib.load(fixed_image_path)
    warped_img = nib.load(warped_image_path)
    fixed_data = fixed_img.get_fdata()
    warped_data = warped_img.get_fdata()
    
    # Extract patient ID and slice information from filenames
    fixed_filename = os.path.basename(fixed_image_path)
    warped_filename = os.path.basename(warped_image_path)
    
    # Get slice ranges from filenames if available
    fixed_range = None
    warped_range = None
    
    # Parse fixed range
    if '_' in fixed_filename:
        parts = fixed_filename.replace('.nii.gz', '').split('_')
        if len(parts) >= 3:
            try:
                f_start = int(parts[-2])
                f_end = int(parts[-1].replace('.nii.gz', ''))
                fixed_range = (f_start, f_end)
                print(f"Fixed slice range from filename: {fixed_range}")
            except (IndexError, ValueError):
                fixed_range = None
    
    # Parse warped range
    if '_' in warped_filename:
        parts = warped_filename.replace('_warped.nii.gz', '').split('_')
        if len(parts) >= 3:
            try:
                w_start = int(parts[-2])
                w_end = int(parts[-1])
                warped_range = (w_start, w_end)
                print(f"Warped slice range from filename: {warped_range}")
            except (IndexError, ValueError):
                warped_range = None
    
    # Check dimensions
    fixed_shape = fixed_data.shape
    warped_shape = warped_data.shape
    
    print(f"Fixed image shape: {fixed_shape}")
    print(f"Warped image shape: {warped_shape}")
    
    # Check if dimensions are compatible
    if fixed_shape[0] != warped_shape[0] or fixed_shape[1] != warped_shape[1]:
        print(f"WARNING: Fixed and warped images have different X-Y dimensions.")
        print(f"Fixed: {fixed_shape[0]}x{fixed_shape[1]}, Warped: {warped_shape[0]}x{warped_shape[1]}")
        print(f"Skipping this pair as they cannot be directly compared.")
        return
    
    # Align and adjust slices based on filename range information
    if fixed_range and warped_range:
        # Determine the overlap between two ranges
        overlap_start = max(fixed_range[0], warped_range[0])
        overlap_end = min(fixed_range[1], warped_range[1])
        
        if overlap_start <= overlap_end:  # There is an overlap
            print(f"Slice overlap range: {overlap_start}-{overlap_end}")
            
            # Calculate the slice indices for both volumes
            fixed_start_idx = overlap_start - fixed_range[0]
            fixed_end_idx = fixed_start_idx + (overlap_end - overlap_start)
            
            warped_start_idx = overlap_start - warped_range[0]
            warped_end_idx = warped_start_idx + (overlap_end - overlap_start)
            
            # Ensure indices are within bounds
            fixed_start_idx = max(0, min(fixed_start_idx, fixed_shape[2] - 1))
            fixed_end_idx = max(0, min(fixed_end_idx, fixed_shape[2] - 1))
            warped_start_idx = max(0, min(warped_start_idx, warped_shape[2] - 1))
            warped_end_idx = max(0, min(warped_end_idx, warped_shape[2] - 1))
            
            # Ensure we have the same number of slices
            slice_count = min(fixed_end_idx - fixed_start_idx + 1, warped_end_idx - warped_start_idx + 1)
            
            if slice_count > 0:
                fixed_end_idx = fixed_start_idx + slice_count - 1
                warped_end_idx = warped_start_idx + slice_count - 1
                
                # Extract the slices of interest
                fixed_data = fixed_data[:, :, fixed_start_idx:fixed_end_idx+1]
                warped_data = warped_data[:, :, warped_start_idx:warped_end_idx+1]
                
                print(f"Using aligned slices - Fixed: {fixed_start_idx}-{fixed_end_idx}, Warped: {warped_start_idx}-{warped_end_idx}")
            else:
                print("WARNING: No overlapping slices found despite range information")
                # Fall back to default behavior
                min_slices = min(fixed_shape[2], warped_shape[2])
                fixed_data = fixed_data[:, :, :min_slices]
                warped_data = warped_data[:, :, :min_slices]
        else:
            print("WARNING: No overlap between fixed and warped slice ranges")
            # Fall back to default behavior
            min_slices = min(fixed_shape[2], warped_shape[2])
            fixed_data = fixed_data[:, :, :min_slices]
            warped_data = warped_data[:, :, :min_slices]
    else:
        # Default behavior if no range information available
        min_slices = min(fixed_shape[2], warped_shape[2])
        if fixed_shape[2] != warped_shape[2]:
            print(f"NOTE: Fixed and warped have different number of slices: {fixed_shape[2]} vs {warped_shape[2]}")
            print(f"Using only the first {min_slices} slices for comparison")
        
        # Trim both arrays to have the same number of slices
        fixed_data = fixed_data[:, :, :min_slices]
        warped_data = warped_data[:, :, :min_slices]
    
    # Generate boundaries
    fixed_boundary = generate_boundary(fixed_data)
    warped_boundary = generate_boundary(warped_data)
    
    # Create a dictionary of boundaries
    boundary_dict = {
        'fixed': fixed_boundary,
        'warped': warped_boundary
    }
    
    # Show animation if requested
    if show_animation:
        create_3d_animation_multiple_boundaries(
            fixed_data, 
            boundary_dict, 
            title=f"Fixed (Blue) vs Warped (Red) - {case_name}"
        )
    
    # Save boundary images if save_path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        save_boundary_images(fixed_data, boundary_dict, case_name, save_path)
        print(f"Saved boundary comparison for {case_name} to {save_path}")

#########################################
# Main Function
#########################################

def main():
    # Directory setup - adjust these paths to your actual directories
    BASE_DIR = '/home/kevin/Desktop/image registration/MIND'
    FIXED_IMAGE_DIR = os.path.join(BASE_DIR, 'fixed')  # Fixed images are in the fixed subdirectory
    RESULT_DIR = os.path.join(BASE_DIR, 'result')  # Warped images are in the result subdirectory
    BOUNDARY_RESULT_DIR = os.path.join(BASE_DIR, 'boundary_result')
    
    # Make sure output directory exists
    os.makedirs(BOUNDARY_RESULT_DIR, exist_ok=True)
    
    # Get all fixed images in the directory
    fixed_files = [f for f in os.listdir(FIXED_IMAGE_DIR) if f.endswith('.nii.gz')]
    
    # Get all warped images
    warped_files = [f for f in os.listdir(RESULT_DIR) if f.endswith('_warped.nii.gz')]
    
    print(f"Found {len(fixed_files)} fixed images and {len(warped_files)} warped images")
    
    # Find matching pairs based on patient name
    processed_pairs = 0
    successful_pairs = 0
    
    # Get all unique patient names
    patient_names = set()
    for fixed_filename in fixed_files:
        if '_' in fixed_filename:
            patient_name = fixed_filename.split('_')[0].strip()
            patient_names.add(patient_name)
    
    print(f"\nFound {len(patient_names)} unique patients")
    
    # Process all patients automatically
    selected_patients = sorted(patient_names)
    
    # Process only the selected patients
    for fixed_filename in fixed_files:
        # Extract patient name from fixed filename (everything before the first underscore)
        if '_' in fixed_filename:
            patient_name = fixed_filename.split('_')[0].strip()
            
            # Skip if not in selected patients
            if patient_name not in selected_patients:
                continue
            
            # Get the slice range from the fixed file
            fixed_range = None
            if len(fixed_filename.split('_')) >= 3:
                try:
                    range_start = fixed_filename.split('_')[-2]
                    range_end = fixed_filename.split('_')[-1].replace('.nii.gz', '')
                    if range_start.isdigit() and range_end.isdigit():
                        fixed_range = (int(range_start), int(range_end))
                except (IndexError, ValueError):
                    fixed_range = None
            
            # Look for warped files containing the same patient name
            matching_warped = [w for w in warped_files if patient_name in w and w.endswith('_warped.nii.gz')]
            
            if matching_warped:
                # Try to find the best matching file based on slice range if available
                best_match = matching_warped[0]  # Default to first
                
                if fixed_range:
                    # Try to find a warped file with overlapping slice range
                    for warped_file in matching_warped:
                        warped_range = None
                        parts = warped_file.replace('_warped.nii.gz', '').split('_')
                        if len(parts) >= 3:
                            try:
                                w_start = parts[-2]
                                w_end = parts[-1]
                                if w_start.isdigit() and w_end.isdigit():
                                    warped_range = (int(w_start), int(w_end))
                            except (IndexError, ValueError):
                                continue
                        
                        if warped_range:
                            # Check for overlap
                            if (warped_range[0] <= fixed_range[1] and warped_range[1] >= fixed_range[0]):
                                best_match = warped_file
                                break
                
                warped_filename = best_match
                fixed_path = os.path.join(FIXED_IMAGE_DIR, fixed_filename)
                warped_path = os.path.join(RESULT_DIR, warped_filename)
                
                # Get a readable case name for the output
                case_name = f"{patient_name}_{fixed_filename.replace('.nii.gz', '')}_vs_{warped_filename.replace('_warped.nii.gz', '')}"
                
                print(f"\nProcessing:\n  Fixed: {fixed_filename}\n  Warped: {warped_filename}")
                
                # Compare the fixed and warped images
                try:
                    compare_fixed_warped(
                        fixed_path,
                        warped_path,
                        case_name,
                        save_path=BOUNDARY_RESULT_DIR,
                        show_animation=True
                    )
                    successful_pairs += 1
                except Exception as e:
                    print(f"Error processing this pair: {str(e)}")
                
                processed_pairs += 1
    
    print(f"\nCompleted processing {processed_pairs} image pairs ({successful_pairs} successful)")

if __name__ == "__main__":
    main()

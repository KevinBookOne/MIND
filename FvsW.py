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

        # (1) Rescale intensities
        p2, p98 = np.percentile(slice_data, (2, 98))
        slice_data = rescale_intensity(slice_data, in_range=(p2, p98), out_range=(0, 1))

        # (2) Slight Gaussian smoothing
        smoothed_data = gaussian_filter(slice_data, sigma=0.5)

        # (3) Otsu threshold
        if np.all(smoothed_data == 0):
            mask = np.zeros_like(smoothed_data, dtype=bool)
        else:
            t = threshold_otsu(smoothed_data)
            mask = smoothed_data > t

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
    
    # Load fixed and warped data
    fixed_data = load_nifti_image(fixed_image_path)
    warped_data = load_nifti_image(warped_image_path)
    
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
    # DeepReg specific directory paths
    FIXED_IMAGE_DIR = "/home/kevin/Desktop/image registration/MIND/fixed"
    RESULT_DIR = "/home/kevin/Desktop/image registration/MIND/result"
    BOUNDARY_RESULT_DIR = "/home/kevin/Desktop/image registration/MIND/boundary_result"
    
    # Make sure output directory exists
    os.makedirs(BOUNDARY_RESULT_DIR, exist_ok=True)
    
    # Get all fixed images in the directory
    fixed_files = [f for f in os.listdir(FIXED_IMAGE_DIR) if f.endswith('.nii.gz') and not f.endswith('_warped.nii.gz')]
    
    for filename in fixed_files:
        case_name = filename.replace('.nii.gz', '')
        
        # The warped image is in the result directory
        warped_filename = f"{case_name}_warped.nii.gz"
        warped_path = os.path.join(RESULT_DIR, warped_filename)
        
        if os.path.exists(warped_path):
            fixed_path = os.path.join(FIXED_IMAGE_DIR, filename)
            
            # Compare the fixed and warped images
            compare_fixed_warped(
                fixed_path,
                warped_path,
                case_name,
                save_path=BOUNDARY_RESULT_DIR,
                show_animation=True
            )

if __name__ == "__main__":
    main()

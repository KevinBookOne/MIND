import os
import numpy as np
import nibabel as nib
import pandas as pd
import cv2
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def load_nii(file_path):
    """Load a NIfTI file and return a NumPy array."""
    return nib.load(file_path).get_fdata()

def preprocess_3d_image(image):
    if len(image.shape) == 3:
        return image
    else:
        raise ValueError(f"Invalid image shape: {image.shape}, expected (D, H, W)")

def threshold_image_manual(image_3d, threshold_value=1):
    mask_3d = (image_3d > threshold_value).astype(np.uint8)
    return mask_3d, threshold_value

def calculate_dice_2d(sliceA, sliceB):
    TP = np.sum((sliceA == 1) & (sliceB == 1))
    A_total = np.sum(sliceA == 1)
    B_total = np.sum(sliceB == 1)
    return 2.0 * TP / (A_total + B_total + 1e-4)

def save_slice_as_png(slice_2d, out_path):
    slice_255 = (slice_2d * 255).astype(np.uint8)
    cv2.imwrite(out_path, slice_255)

def save_DI_result_excel(results, excel_filename="DI_result.xlsx"):
    df = pd.DataFrame(results)
    df.to_excel(excel_filename, index=False)
    print(f"✅ DI results saved to {excel_filename}")

def extract_case_name(warped_name):
    """Extracts the correct case name from the warped filename by removing the extra iteration number if present.
    Expected pattern: <name>_<num1>_<num2> (optional _<iters>)_warped.nii[.gz]
    Returns: <name>_<num1>_<num2>_warped.nii[.gz]"""
    return warped_name.replace('_warped.nii.gz', '')

def get_file_pairs(fixed_dir, moving_dir):
    """Get matching file pairs from fixed and moving directories.
    Returns a list of tuples (fixed_path, moving_path, case_name)."""
    fixed_files = {}
    moving_files = {}

    fixed_filenames = [f for f in os.listdir(fixed_dir) if f.endswith(".nii.gz")]
    moving_filenames = [f for f in os.listdir(moving_dir) if f.endswith(".nii.gz")]

    print(f"🔍 固定影像數量: {len(fixed_filenames)}，移動影像數量: {len(moving_filenames)}")

    for filename in fixed_filenames:
        case_name = filename.replace(".nii.gz", "")  # 保留完整檔名
        fixed_files[case_name] = os.path.join(fixed_dir, filename)

    for filename in moving_filenames:
        case_name = filename.replace(".nii.gz", "")  # 保留完整檔名
        moving_files[case_name] = os.path.join(moving_dir, filename)

    file_pairs = []
    for case_name in fixed_files.keys():
        if case_name in moving_files:
            file_pairs.append((fixed_files[case_name], moving_files[case_name], case_name))

    print(f"✅ 匹配的測資數量: {len(file_pairs)}")
    return file_pairs

def evaluate_3d_dice(fixed_dir, moving_dir, warped_dir, output_dir, patient_dirs=None):
    """Evaluate 3D Dice scores between fixed-moving and fixed-warped image pairs.
    
    Args:
        fixed_dir (str): Directory containing fixed images (used in legacy mode)
        moving_dir (str): Directory containing moving images (used in legacy mode)
        warped_dir (str): Directory containing warped images (used in legacy mode)
        output_dir (str): Directory to save evaluation results
        patient_dirs (list): List of patient directories to process (new multi-case mode)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "dice_images"), exist_ok=True)
    
    fw_results = []  # Collect Fixed vs Warped Dice results

    print("Filename\tMaxDice_FM\tSlice_FM\tMaxDice_FW\tSlice_FW")
    
    # Determine if we're in legacy mode or multi-case mode
    if patient_dirs:
        # Multi-case mode - process patient directories
        for patient_dir in patient_dirs:
            patient_name = os.path.basename(patient_dir)
            print(f"\nProcessing patient directory: {patient_name}")
            
            # Locate the files in the patient directory
            try:
                # Read case_info.txt to find the filenames
                case_info = {}
                case_info_path = os.path.join(patient_dir, "case_info.txt")
                
                if os.path.exists(case_info_path):
                    with open(case_info_path, 'r') as f:
                        for line in f:
                            key, value = line.strip().split('=', 1)
                            case_info[key] = value
                    
                    fixed_path = case_info.get('fixed_file')
                    moving_path = case_info.get('moving_file')
                    case_name = case_info.get('case_name', patient_name)
                    warped_name = f"{case_name}_warped.nii.gz"
                    warped_path = os.path.join("warped", warped_name)
                else:
                    # If no case_info.txt, look directly in the images folder
                    images_dir = os.path.join(patient_dir, "images")
                    if os.path.exists(images_dir):
                        image_files = [f for f in os.listdir(images_dir) if f.endswith(".nii.gz")]
                        if len(image_files) >= 2:
                            # Assume first file is fixed, second is moving
                            fixed_path = os.path.join(images_dir, image_files[0])
                            moving_path = os.path.join(images_dir, image_files[1])
                            case_name = patient_name
                            warped_name = f"{case_name}_warped.nii.gz"
                            warped_path = os.path.join("warped", warped_name)
                        else:
                            print(f"[Warning] Not enough image files in {images_dir}, skipping.")
                            continue
                    else:
                        print(f"[Warning] No images directory found in {patient_dir}, skipping.")
                        continue

                if not os.path.exists(fixed_path) or not os.path.exists(moving_path):
                    print(f"[Warning] Fixed or moving image not found for {case_name}, skipping.")
                    continue
                    
                if not os.path.exists(warped_path):
                    print(f"[Warning] Warped image not found: {warped_path}, skipping.")
                    continue
                
                process_case_dice(fixed_path, moving_path, warped_path, case_name, output_dir, fw_results)
                
            except Exception as e:
                print(f"[Error] Processing failed for {patient_name}: {e}")
    else:
        # Legacy mode - process files from fixed/moving directories
        file_pairs = get_file_pairs(fixed_dir, moving_dir)
        if not file_pairs:
            raise ValueError("No matching fixed and moving image pairs found.")
            
        for fixed_path, moving_path, case_name in file_pairs:
            try:
                warped_name = f"{case_name}_warped.nii.gz"
                warped_path = os.path.join(warped_dir, warped_name)
                
                if not os.path.exists(warped_path):
                    print(f"[Warning] Warped image not found: {warped_name}, skipping.")
                    continue
                    
                print(f"\n--- Processing case: {case_name} ---")
                print(f"  fixed:   {fixed_path}")
                print(f"  moving:  {moving_path}")
                print(f"  warped:  {warped_path}")

                # Process this case
                process_case_dice(fixed_path, moving_path, warped_path, case_name, output_dir, fw_results)
            except Exception as e:
                print(f"[Error] Processing failed for {case_name}: {e}")

    print("✅ Completed slice-by-slice evaluation.")

    # Sort in lexicographical order by filename
    fw_results_sorted = sorted(fw_results, key=lambda x: x["Filename"])
    excel_path = os.path.join(output_dir, "DI_result.xlsx")
    save_DI_result_excel(fw_results_sorted, excel_filename=excel_path)
    print("\n==== SUMMARY OF DICE EVALUATION RESULTS ====\n")
    # Create a formatted table header
    print(f"{'Case Name':<30} {'Before (FM)':<15} {'After (FW)':<15} {'Change':<10}")
    print("-" * 70)
    
    # Print each result with improvement
    for result in fw_results_sorted:
        before = result["MaxDice_FM"]
        after = result["MaxDice_FW"]
        improvement = (after - before) / before * 100 if before > 0 else float('inf')
        improvement_str = f"+{improvement:.2f}%" if improvement > 0 else f"{improvement:.2f}%"
        
        print(f"{result['Filename']:<30} {before:.4f}       {after:.4f}       {improvement_str}")

def process_case_dice(fixed_path, moving_path, warped_path, case_name, output_dir, fw_results):
    """Process a single case for Dice score evaluation"""
    print(f"\n--- Processing case: {case_name} ---")
    print(f"  fixed:   {fixed_path}")
    print(f"  moving:  {moving_path}")
    print(f"  warped:  {warped_path}")
    
    # Load volumes
    fixed_image  = preprocess_3d_image(load_nii(fixed_path))
    moving_image = preprocess_3d_image(load_nii(moving_path))
    warped_image = preprocess_3d_image(load_nii(warped_path))

    # Threshold using fixed threshold of 170 (0-255 scale)
    fixed_bin,  _ = threshold_image_manual(fixed_image, threshold_value=1)
    moving_bin, _ = threshold_image_manual(moving_image, threshold_value=1)
    warped_bin, _ = threshold_image_manual(warped_image, threshold_value=1)

    depth = fixed_bin.shape[2]
    dice_slice_fm = []
    dice_slice_fw = []

    for s in range(depth):
        slice_fixed  = fixed_bin[:, :, s]
        slice_moving = moving_bin[:, :, s]
        slice_warped = warped_bin[:, :, s]

        dice2d_fm = calculate_dice_2d(slice_fixed, slice_moving)
        dice_slice_fm.append(dice2d_fm)

        dice2d_fw = calculate_dice_2d(slice_fixed, slice_warped)
        dice_slice_fw.append(dice2d_fw)

    max_fm = max(dice_slice_fm)
    idx_fm = np.argmax(dice_slice_fm)
    max_fw = max(dice_slice_fw)
    idx_fw = np.argmax(dice_slice_fw)

    # Calculate improvement percentage
    improvement = (max_fw - max_fm) / max_fm * 100 if max_fm > 0 else float('inf')
    improvement_str = f"+{improvement:.2f}%" if improvement > 0 else f"{improvement:.2f}%"
    
    print(f"\n==== DICE SCORE RESULTS: {case_name} ====\n")
    print(f"BEFORE registration (Fixed vs Moving):    {max_fm:.4f} (slice {idx_fm})")
    print(f"AFTER registration (Fixed vs Warped):     {max_fw:.4f} (slice {idx_fw})")
    print(f"Improvement:                              {improvement_str}\n")

    # Create case-specific output directory
    case_output_dir = os.path.join(output_dir, "dice_images", case_name)
    os.makedirs(case_output_dir, exist_ok=True)

    best_slice_fixed_fm  = fixed_bin[:, :, idx_fm]
    best_slice_moving_fm = moving_bin[:, :, idx_fm]
    best_slice_fixed_fw  = fixed_bin[:, :, idx_fw]
    best_slice_warped_fw = warped_bin[:, :, idx_fw]

    # Save best slices as PNG
    save_slice_as_png(best_slice_fixed_fm,
                      os.path.join(case_output_dir, f"{case_name}_bestFM_fixed_{idx_fm}.png"))
    save_slice_as_png(best_slice_moving_fm,
                      os.path.join(case_output_dir, f"{case_name}_bestFM_moving_{idx_fm}.png"))
    save_slice_as_png(best_slice_fixed_fw,
                      os.path.join(case_output_dir, f"{case_name}_bestFW_fixed_{idx_fw}.png"))
    save_slice_as_png(best_slice_warped_fw,
                      os.path.join(case_output_dir, f"{case_name}_bestFW_warped_{idx_fw}.png"))
            
    # Generate comparison image with matplotlib
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    ax1.imshow(best_slice_fixed_fm, cmap="gray")
    ax1.set_title(f"Fixed - Slice {idx_fm}")
    ax1.axis("off")
    
    ax2.imshow(best_slice_moving_fm, cmap="gray")
    ax2.set_title(f"Moving - Slice {idx_fm} (Dice: {max_fm:.4f})")
    ax2.axis("off")
    
    ax3.imshow(best_slice_fixed_fw, cmap="gray")
    ax3.set_title(f"Fixed - Slice {idx_fw}")
    ax3.axis("off")
    
    ax4.imshow(best_slice_warped_fw, cmap="gray")
    ax4.set_title(f"Warped - Slice {idx_fw} (Dice: {max_fw:.4f})")
    ax4.axis("off")
            
    plt.tight_layout()
    plt.savefig(os.path.join(case_output_dir, f"{case_name}_dice_comparison.png"), dpi=150)
    plt.close(fig)

    # Store results
    fw_results.append({
        "Filename": case_name,
        "MaxDice_FM": max_fm,
        "Slice_FM": idx_fm,
        "MaxDice_FW": max_fw,
        "Slice_FW": idx_fw
    })

    print("✅ Completed slice-by-slice evaluation.")

    # Sort in lexicographical order by filename
    fw_results_sorted = sorted(fw_results, key=lambda x: x["Filename"])
    excel_path = os.path.join(output_dir, "DI_result.xlsx")
    save_DI_result_excel(fw_results_sorted, excel_filename=excel_path)
    print("\n==== SUMMARY OF DICE EVALUATION RESULTS ====\n")
    # Create a formatted table header
    print(f"{'Case Name':<30} {'Before (FM)':<15} {'After (FW)':<15} {'Change':<10}")
    print("-" * 70)
    
    # Print each result with improvement
    for result in fw_results_sorted:
        before = result["MaxDice_FM"]
        after = result["MaxDice_FW"]
        improvement = (after - before) / before * 100 if before > 0 else float('inf')
        improvement_str = f"+{improvement:.2f}%" if improvement > 0 else f"{improvement:.2f}%"
        
        print(f"{result['Filename']:<30} {before:.4f}       {after:.4f}       {improvement_str}")

def process_simple_output(output_dir='output_dir'):
    """Process outputs from SIMPLE.py."""
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Directory '{output_dir}' not found. Make sure SIMPLE.py ran successfully.")
    print("Processing outputs from SIMPLE.py in", output_dir)

def find_patient_directories(root_dir="."):
    """Find patient directories that contain the required files.
    A valid patient directory should contain:
    - case_info.txt file, OR
    - an images directory with .nii.gz files
    """
    patient_dirs = []
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            # Check if this is a patient directory with required files
            case_info_path = os.path.join(item_path, "case_info.txt")
            images_dir = os.path.join(item_path, "images")
            
            if os.path.exists(case_info_path) or (os.path.exists(images_dir) and any(f.endswith(".nii.gz") for f in os.listdir(images_dir))):
                patient_dirs.append(item_path)
    
    return patient_dirs

def main():
    # Define base paths
    base_dir = '/home/kevin/Desktop/image registration/MIND'
    
    # Legacy mode paths
    legacy_fixed_dir = os.path.join(base_dir, 'fixed')
    legacy_moving_dir = os.path.join(base_dir, 'moving')
    legacy_warped_dir = os.path.join(base_dir, 'result')  # Changed from 'warped' to 'result'
    output_dir = os.path.join(base_dir, 'output_dice')
    
    # Check if legacy mode directories exist
    legacy_mode = os.path.exists(legacy_fixed_dir) and os.path.exists(legacy_moving_dir)
    
    # Check for multi-patient directories
    patient_dirs = find_patient_directories(base_dir)
    
    print(f"Found {len(patient_dirs)} patient directories")
    
    if legacy_mode:
        print("Running in legacy mode - processing files from fixed/moving directories")
        evaluate_3d_dice(legacy_fixed_dir, legacy_moving_dir, legacy_warped_dir, output_dir)
    elif patient_dirs:
        print("Running in multi-case mode - processing patient directories")
        evaluate_3d_dice(None, None, None, output_dir, patient_dirs=patient_dirs)
    else:
        print("No valid directories found to process")

if __name__ == "__main__":
    main()
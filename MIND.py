import os
import numpy as np
import shutil
import time
import datetime
import nibabel as nib
import tensorflow as tf
from tqdm import tqdm
import deepreg.model.layer_util as layer_util
from deepreg.registry import REGISTRY

# Import the custom MIND loss functions from mind_loss.py
from mind_loss import mind_loss

# Record the start time
start_time = time.time()
start_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Initialize a list to store time records for each case
case_time_records = []

# ============================
# 1. Registration Setup & Loss
# ============================

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

# Training parameters
learning_rate = 0.01
total_iter = 100 

# ==========================================
# 2. File Pairing Function for Registration
# ==========================================
def get_file_pairs(fixed_dir, moving_dir):
    """
    Improved get_file_pairs that ensures all `_start_end.nii.gz` files match.
    """
    fixed_files = {}
    moving_files = {}

    fixed_filenames = [f for f in os.listdir(fixed_dir) if f.endswith(".nii.gz")]
    moving_filenames = [f for f in os.listdir(moving_dir) if f.endswith(".nii.gz")]

    print(f"🔍 固定影像數量: {len(fixed_filenames)}，移動影像數量: {len(moving_filenames)}")

    for filename in fixed_filenames:
        case_name = filename.replace(".nii.gz", "")
        fixed_files[case_name] = os.path.join(fixed_dir, filename)

    for filename in moving_filenames:
        case_name = filename.replace(".nii.gz", "")
        moving_files[case_name] = os.path.join(moving_dir, filename)

    file_pairs = []
    for case_name in fixed_files.keys():
        if case_name in moving_files:
            file_pairs.append((fixed_files[case_name], moving_files[case_name], case_name))

    print(f"✅ 匹配的測資數量: {len(file_pairs)}")
    return file_pairs

def train_step(grid, weights, optimizer, mov, fix):
    """
    Single training step using the MIND-based loss.
    """
    with tf.GradientTape() as tape:
        pred = layer_util.resample(vol=mov, loc=layer_util.warp_grid(grid, weights))
        loss = mind_loss(fix, pred)
    gradients = tape.gradient(loss, [weights])
    optimizer.apply_gradients(zip(gradients, [weights]))
    return loss.numpy()

# ================================
# 3. Global Paths & Preparation
# ================================
FIXED_IMAGE_DIR = "/home/kevin/Desktop/image registration/MIND/fixed"
MOVING_IMAGE_DIR = "/home/kevin/Desktop/image registration/MIND/moving"
SAVE_PATH = "/home/kevin/Desktop/image registration/MIND/result"

# Ensure output directory exists
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

# Get paired file paths
file_pairs = get_file_pairs(FIXED_IMAGE_DIR, MOVING_IMAGE_DIR)
if not file_pairs:
    raise ValueError("No matching fixed and moving image pairs found.")



# ============================================
# 4. Registration, Rescaling, and Conversion
# ============================================
for fixed_image_path, moving_image_path, case_name in file_pairs:
    # Record the case start time
    case_start_time = time.time()
    case_start_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Extract start and end indexes from case_name if applicable
    parts = case_name.split("_")
    if len(parts) >= 2:
        start_idx, end_idx = parts[-2:]
    else:
        start_idx = end_idx = "NA"
    
    # Output path for the warped NIfTI file
    output_nifti_path = os.path.join(SAVE_PATH, f"{case_name}_warped.nii.gz")
    
    # Skip already processed cases
    if os.path.exists(output_nifti_path):
        tqdm.write(f"🚀 {case_name} 已處理過，跳過此影像。")
        continue

    print(f"\n📌 處理: {case_name} (start:{start_idx}, end:{end_idx})")
    t0 = time.time()
    fixed_image_data = nib.load(fixed_image_path).get_fdata()
    moving_image_data = nib.load(moving_image_path).get_fdata()
    t1 = time.time()
    print(f"🕒 影像載入時間: {t1 - t0:.2f} 秒")

    # Record original intensity ranges
    fix_min_val = fixed_image_data.min()
    fix_max_val = fixed_image_data.max()
    mov_min_val = moving_image_data.min()
    mov_max_val = moving_image_data.max()

    # Convert images to tensors and add a batch dimension
    fixed_image = tf.convert_to_tensor(fixed_image_data, dtype=tf.float32)[None, ..., None]
    moving_image = tf.convert_to_tensor(moving_image_data, dtype=tf.float32)[None, ..., None]
    # Normalize images to [0,1]
    fixed_image = (fixed_image - fix_min_val) / (fix_max_val - fix_min_val + 1e-8)
    moving_image = (moving_image - mov_min_val) / (mov_max_val - mov_min_val + 1e-8)

    # Initialize training parameters
    grid_ref = layer_util.get_reference_grid(grid_size=fixed_image.shape[1:4])
    var_affine = tf.Variable(
        initial_value=[[[1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0]]],
        trainable=True,
    )
    optimiser = tf.optimizers.SGD(learning_rate)

    # Training loop with progress output every 10 steps
    for step in tqdm(range(total_iter), desc=f"Training {case_name}", ncols=80):
        s0 = time.time()
        loss_value = train_step(grid_ref, var_affine, optimiser, moving_image, fixed_image)
        s1 = time.time()
        if (step + 1) % 10 == 0 or step == 0:
            tqdm.write(f"Step {step + 1}: Loss = {loss_value:.5f} 🕒 {(s1 - s0):.3f} 秒")

    print(f"✅ 訓練完成 {case_name}，進入下一個 case...\n")
    grid_opt = layer_util.warp_grid(grid_ref, var_affine)
    warped_moving_image = layer_util.resample(vol=moving_image, loc=grid_opt)
    
    # Rescale the warped image back to the original moving image intensity range
    scaled_warped_image = warped_moving_image * (mov_max_val - mov_min_val) + mov_min_val

    try:
        # Squeeze out the batch dimension and save as NIfTI
        nib.save(
            nib.Nifti1Image(tf.squeeze(scaled_warped_image).numpy(), affine=np.eye(4)),
            output_nifti_path
        )
        print(f"✅ 已存入 (已 scale 回原始範圍): {output_nifti_path}")

        
        # Record case end time and calculate duration
        case_end_time = time.time()
        case_end_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        case_duration = case_end_time - case_start_time
        
        # Store case timing information
        case_time_records.append({
            "case_name": case_name,
            "start_time": case_start_datetime,
            "end_time": case_end_datetime,
            "duration_seconds": case_duration,
            "duration_minutes": case_duration / 60
        })
        
        print(f"⏱️ 處理時間: {case_duration:.2f} 秒 ({case_duration/60:.2f} 分鐘)")
    except Exception as e:
        print(f"🚨 儲存失敗: {case_name}，錯誤訊息: {e}")
        continue



# Record the end time
end_time = time.time()
end_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Calculate time duration
registration_duration = end_time - start_time

# Format time information with per-case details
time_info = f"Registration Time Log\n=====================\nStart time: {start_datetime}\nEnd time: {end_datetime}\nTotal duration: {registration_duration:.2f} seconds ({registration_duration/60:.2f} minutes)\n"

# Add detailed time records for each case
time_info += "\nCase-by-Case Processing Times:\n"
time_info += "==============================\n"

for record in case_time_records:
    time_info += f"Case: {record['case_name']}\n"
    time_info += f"  Start: {record['start_time']}\n"
    time_info += f"  End: {record['end_time']}\n"
    time_info += f"  Duration: {record['duration_seconds']:.2f} seconds ({record['duration_minutes']:.2f} minutes)\n\n"

# Save time information to a text file
time_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mind_registration_time_log.txt")
with open(time_log_path, 'w') as f:
    f.write(time_info)

# Also save a summary of all processed cases
f.write("\nAll Processed Cases:\n")
for fixed_path, moving_path, case_name in file_pairs:
    if any(record['case_name'] == case_name for record in case_time_records):
        f.write(f"- {case_name} ✓\n")
    else:
        f.write(f"- {case_name} (skipped)\n")

print(f"✅ Registration finished!! 總處理時間: {registration_duration:.2f} 秒")
print(f"✅ 時間記錄已儲存至: {time_log_path}")

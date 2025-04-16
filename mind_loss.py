import tensorflow as tf

def gaussian_kernel_3d(patch_size, sigma):
    """
    Creates a 3D Gaussian kernel.
    
    Parameters:
        patch_size (int): Size of the kernel (assumed cubic).
        sigma (float): Standard deviation of the Gaussian.
    
    Returns:
        A 5D tensor of shape (patch_size, patch_size, patch_size, 1, 1) 
        representing the Gaussian kernel.
    """
    # Create a coordinate grid. Note: range is set to ensure symmetric kernel.
    ax = tf.range(-patch_size // 2 + 1, patch_size // 2 + 1, dtype=tf.float32)
    xx, yy, zz = tf.meshgrid(ax, ax, ax, indexing='ij')
    kernel = tf.exp(-(xx**2 + yy**2 + zz**2) / (2.0 * sigma**2))
    # Normalize the kernel so that the sum equals 1.
    kernel = kernel / tf.reduce_sum(kernel)
    # Reshape to match conv3d filter dimensions.
    kernel = tf.reshape(kernel, (patch_size, patch_size, patch_size, 1, 1))
    return kernel

def compute_MIND_descriptor(
    image,
    patch_size=3,
    offsets=[
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ],
    alpha=1.0,
    eps=1e-5
):
    """
    Modified MIND descriptor that:
      1) Computes the offset-based patch SSD for each offset.
      2) Averages the SSDs across offsets to obtain local_var.
      3) Uses that average for the exponential map.

    Parameters:
        image: 5D tensor [batch, D, H, W, channels]
        patch_size: size of the Gaussian patch window
        offsets: list of 3D offsets (tuples)
        alpha: scale factor for local variance
        eps: small constant to avoid division by zero
    
    Returns:
        mind_descriptor: a 5D tensor [batch, D, H, W, num_offsets]
    """
    # Pad the image to accommodate the patch neighborhood
    pad = patch_size // 2
    padded = tf.pad(
        image,
        paddings=[[0, 0], [pad, pad], [pad, pad], [pad, pad], [0, 0]],
        mode='REFLECT'
    )
    
    # Create Gaussian kernel for computing patch-based SSD
    sigma = 0.5
    kernel_gaussian = gaussian_kernel_3d(patch_size, sigma)
    
    # Step 1: For each offset, compute the patch SSD
    offset_ssds = []  # will store the patch SSD for each offset
    for offset in offsets:
        # shift the padded image
        shifted = tf.roll(padded, shift=offset, axis=[1, 2, 3])
        # voxel-wise squared difference
        diff = tf.square(padded - shifted)
        # Gaussian-weighted sum of squared differences
        patch_ssd = tf.nn.conv3d(diff, kernel_gaussian,
                                 strides=[1, 1, 1, 1, 1],
                                 padding='VALID')
        offset_ssds.append(patch_ssd)

    # Stack them along a new dimension for offsets
    # shape: [batch, D, H, W, num_offsets]
    offset_ssds_all = tf.stack(offset_ssds, axis=-1)

    # Step 2: Compute local_var = average SSD across offsets (the self-similarity variance)
    # shape: [batch, D, H, W, 1]
    local_var = tf.reduce_mean(offset_ssds_all, axis=-1, keepdims=True)

    # Optional clamp to avoid extremely large/small values
    mean_var = tf.reduce_mean(local_var)
    lower_bound = 0.001 * mean_var
    upper_bound = 1000.0 * mean_var
    local_var = tf.clip_by_value(local_var, lower_bound, upper_bound)

    # Step 3: For each offset, compute the MIND descriptor
    descriptors = []
    # We already have offset_ssds_all with shape [batch, D, H, W, num_offsets]
    num_offsets = len(offsets)
    for i in range(num_offsets):
        # pick out the SSD map for this offset
        ssd_i = offset_ssds_all[..., i:(i+1)]  # keep shape [batch, D, H, W, 1]
        # exponent
        descriptor_i = tf.exp(- ssd_i / (alpha * local_var + eps))
        descriptors.append(descriptor_i)

    # Concatenate along the last dimension => final shape: [batch, D, H, W, num_offsets]
    mind_descriptor = tf.concat(descriptors, axis=-1)
    return mind_descriptor

def mind_loss(
    fixed_image,
    moving_image_warped,
    patch_size=3,
    offsets=[
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ]
):
    """
    Compute the MIND-based loss between the fixed image and the warped moving image,
    using the offset-based local variance definition.

    Returns a mean-squared difference between MIND descriptors.
    """
    descriptor_fixed = compute_MIND_descriptor(fixed_image, patch_size, offsets)
    descriptor_moving = compute_MIND_descriptor(moving_image_warped, patch_size, offsets)
    # sum of the square of the difference, then mean
    loss = tf.reduce_mean(tf.square(descriptor_fixed - descriptor_moving))
    return loss

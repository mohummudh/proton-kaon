# TODO: topology + calorimetry similarity + retrieval (maybe a seperate file)

import numpy as np
import torch
from scipy.ndimage import zoom


def cut_start(image, target=50):
    """
    Selects only the last {target} no. of rows (wires) of an image.
    """
    return image[-target:, :]

def pad_image(image, target_wh=(1502, 179)):
    """
    Place a 2D image block in the centre of a zero canvas.

    image: 2D np.ndarray, shape (h, w)
    target_wh: (target_width, target_height)
    """
    target_w, target_h = target_wh
    canvas = np.zeros((target_h, target_w), dtype=image.dtype)

    h, w = image.shape
    
    v = int(np.argmax(image[0]))

    a = 751 - v
    
    # Compute top-left corner so image is centred, clamped to fit within canvas
    y0 = 0
    x0 = max(0, min(a, target_w - w))  # Clamp to valid range [0, target_w - w]
    y1 = y0 + h
    x1 = x0 + w

    # Place the image block
    canvas[y0:y1, x0:x1] = image

    return canvas

def pad_image_batch_gpu(images_list, target_wh=(1502, 51), device='cuda', batch_size=32, cut_rows=None):
    """
    GPU-accelerated batch padding of images.
    
    images_list: list of 2D np.ndarrays
    target_wh: (target_width, target_height)
    device: 'cuda' or 'cpu'
    batch_size: number of images to process at once
    """
    target_w, target_h = target_wh
    results = []
    
    for batch_start in range(0, len(images_list), batch_size):
        batch_end = min(batch_start + batch_size, len(images_list))
        batch_images = images_list[batch_start:batch_end]

        if cut_rows is not None:
            batch_images = [cut_start(img, target=cut_rows) for img in batch_images]
        
        # Find max dimensions in batch
        max_h = max(img.shape[0] for img in batch_images)
        max_w = max(img.shape[1] for img in batch_images)
        
        # Create padded batch on GPU
        padded_batch = torch.zeros(
            len(batch_images), target_h, target_w,
            dtype=torch.float32, device=device
        )
        
        for i, img in enumerate(batch_images):
            img_tensor = torch.from_numpy(img).float().to(device)
            h, w = img.shape
            
            # Find peak position
            v = torch.argmax(img_tensor[0]).item()
            a = 751 - v
            
            # Compute placement
            y0 = 0
            x0 = max(0, min(a, target_w - w))
            y1 = y0 + h
            x1 = x0 + w
            
            # Place on canvas
            padded_batch[i, y0:y1, x0:x1] = img_tensor
        
        # Move back to CPU and convert to numpy
        results.extend(padded_batch.cpu().numpy())
    
    return results


def downsample_image(image, target_shape=(512, 40)):
    '''
    Reduce an image from original to target shape.
    '''
    scale_y = target_shape[1] / image.shape[0]
    scale_x = target_shape[0] / image.shape[1]
    return zoom(image, (scale_y, scale_x), order=1)  # order=1 for bilinear interpolation

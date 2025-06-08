import torch
from PIL import Image


def load_image(filename, size=None, scale=None):
    """
    Load an image and optionally resize it.

    Args:
        filename (str): Path to the image file.
        size (int, optional): If provided, resize image to (size, size).
        scale (float, optional): If provided and size is None, downscale image
                                 by this factor (i.e., new_width = old_width/scale).

    Returns:
        PIL.Image.Image: The loaded (and resized) image.
    """
    img = Image.open(filename).convert('RGB')
    if size is not None:
        # Use high-quality downsampling filter
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    elif scale is not None:
        new_width = int(img.size[0] / scale)
        new_height = int(img.size[1] / scale)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return img


def save_image(filename, data):
    """
    Save a tensor as an image file.

    Args:
        filename (str): Path where to save the image.
        data (torch.Tensor): Tensor of shape (C, H, W) with values in [0, 255].
    """
    # Clone to avoid modifying original, clamp to valid range, convert to numpy
    img = data.clone().clamp(0, 255).cpu().numpy()
    # Convert from (C, H, W) to (H, W, C)
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def gram_matrix(y):
    """
    Compute the Gram matrix for a batch of feature maps.

    Args:
        y (torch.Tensor): Feature maps of shape (B, C, H, W).

    Returns:
        torch.Tensor: Gram matrices of shape (B, C, C).
    """
    b, ch, h, w = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    # Compute Gram and normalize by number of elements
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    """
    Normalize image batch using ImageNet mean and std.

    Args:
        batch (torch.Tensor): Batch of images of shape (B, C, H, W) with values in [0, 255].

    Returns:
        torch.Tensor: Normalized batch.
    """
    # Move pixel values to [0,1]
    batch = batch.div_(255.0)
    # ImageNet statistics
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std

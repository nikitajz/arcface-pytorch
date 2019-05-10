def clop_center(img, target_shape: tuple):
    """
    crop center of image

    Args:
        img: PIL.Image
        target_shape:

    Returns:
        new Image
    """

    if len(target_shape) != 2:
        raise ValueError(f'target shape length must be 2. {target_shape}')
    new_width, new_height = target_shape
    width, height = img.size

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    return img.crop((left, top, right, bottom))

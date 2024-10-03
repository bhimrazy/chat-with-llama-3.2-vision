import base64
import os
from io import BytesIO
from typing import List

from src.config import IMAGE_EXTENSIONS
from PIL import Image


def get_file_extension(filename):
    return os.path.splitext(filename)[1].lower()


def is_image(filename):
    return get_file_extension(filename) in IMAGE_EXTENSIONS


def all_images(files):
    return all(is_image(file.name) for file in files)


def prepare_content_with_images(content: str, images: List[object]):
    """Prepare content with images."""
    content = [
        *images,
        {
            "type": "text",
            "text": content,
        },
    ]
    return content


def encode_image(image_source):
    """
    Encode an image to a base64 data URL based object.

    Parameters:
    image_source (str or Image): The image source. Can be a real image URL, an Image instance, or a base64 URL string.

    Returns:
    str or None: The base64-encoded data URL of the image if successful, otherwise None.
    """
    try:
        image = Image.open(image_source).convert("RGB")

        # resize to max_size
        max_size = 720
        if max(image.size) > max_size:
            w, h = image.size
            if w > h:
                new_w = max_size
                new_h = int(h * max_size / w)
            else:
                new_h = max_size
                new_w = int(w * max_size / h)
            image = image.resize((new_w, new_h), resample=Image.BICUBIC)

        buffered = BytesIO()
        # Use image format or default to "JPEG"
        image_format = image.format if image.format else "JPEG"
        image.save(buffered, format=image_format)
        mime_type = f"image/{image_format.lower()}"
        encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        url = f"data:{mime_type};base64,{encoded_image}"

        image_object = {
            "type": "image_url",
            "image_url": {
                "url": url,
            },
        }
        return image_object
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

import io
import os
from typing import List, Tuple
from owslib.wms import WebMapService
from PIL import Image
import matplotlib.pyplot as plt

import hashlib

def get_bbox_dataforsyningen(
    bbox: Tuple[float, float, float, float],
    wms: object,
    token: str,
    layers: List[str] = ["geodanmark_2023_12_5cm"],
    size: Tuple[int, int] = None,
    transparent: bool = True,
):
    """
    Get an image from a WMS service based on a bounding box. Results are cached in data/external/wmsCache.

    Parameters:
        bbox (tuple): The bounding box coordinates in the format (minx, miny, maxx, maxy).
        wms (object): The WMS service object.
        token (str): The access token for the WMS service.
        layers (List[str]): The list of layer names to retrieve from the WMS service. Default is ["geodanmark_2023_12_5cm"].
        size (tuple): The size of the image in pixels. Default is (400, 400).
        transparent (bool): Whether the image should have a transparent background. Default is True.

    Returns:
        PIL.Image.Image: The retrieved image.
    """

    # (width, height) approximately 8 pixels per meter as wms resoultion is 12.5 cm/pixel
    if size is None:
        size = int(8 * (bbox[2] - bbox[0])), int(8 * (bbox[3] - bbox[1]))

    # Calculate the hash for the request
    params_string = f"{wms.url}{wms.version}{bbox}{layers}{size}{transparent}"
    request_hash = hashlib.md5(params_string.encode()).hexdigest()
    png_path = f"wmsCache/{request_hash}.png"
    
    if os.path.exists(png_path):
        image = Image.open(png_path)
        return image
    
    else:
        wms_response = wms.getmap(
            layers=layers,
            srs="EPSG:25832",
            bbox=bbox,
            size=size,
            format="image/png",
            transparent=transparent,
            token=token,
        )
        image = Image.open(io.BytesIO(wms_response.read()))
        image.save(png_path)
        return image
    
def format_bbox(
    bbox: Tuple[float, float, float, float], padding: int = 0
    ) -> Tuple[int, int, int, int]:
    """
    Format the bounding box coordinates with padding.

    Args:
        bbox (Tuple[float, float, float, float]): The input bounding box coordinates (xmin, ymin, xmax, ymax).
        padding (float, optional): The padding value to add to each side of the bounding box. Defaults to 0.

    Returns:
        Tuple[float, float, float, float]: The formatted bounding box coordinates with padding.
    """
    bbox = (
        bbox[0] - padding,
        bbox[1] - padding,
        bbox[2] + padding,
        bbox[3] + padding,
    )  # Post-padding
    bbox = tuple(round(coord) for coord in bbox)
    return bbox

if __name__ == "__main__":
    wms_url = "https://api.dataforsyningen.dk/orto_foraar_DAF?service=WMS"
    TOKEN = "f8dbeb10b068e37b646751d5da8ffaaf"
    wms = WebMapService(wms_url, version="1.3.0", timeout=60)
    bbox = (725000, 6170000, 726000, 6171000)
    
    im_size = (
        8 * (bbox[2] - bbox[0]),
        8 * (bbox[3] - bbox[1]),
    )  # (width, height) approximately 8 pixels per meter as WMS resolution is 12.5 cm/pixel
    GISimage = get_bbox_dataforsyningen(bbox, wms, TOKEN, size=im_size)

    plt.imshow(GISimage)
    plt.show()
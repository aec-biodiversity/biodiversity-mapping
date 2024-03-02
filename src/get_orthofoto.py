import io
import os
from typing import List, Tuple
from owslib.wms import WebMapService
from PIL import Image
import matplotlib.pyplot as plt

import hashlib
import requests
import imageio
import numpy as np


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

def get_bbox_dataforsyningen_wcs(
    bbox: Tuple[float, float, float, float],
    token: str,
    layer: str = "dhm_terraen",
    size: Tuple[int, int] = None
):
    # (width, height) approximately 8 pixels per meter as wms resoultion is 12.5 cm/pixel
    if size is None:
        size = int(8 * (bbox[2] - bbox[0])), int(8 * (bbox[3] - bbox[1]))

    url_wcs = "https://api.dataforsyningen.dk/dhm_wcs_DAF?service=WCS&version=1.0"
    params = f"&token={token}&REQUEST=GetCoverage&coverage={layer}&CRS=epsg:25832&bbox={str(bbox)[1:-1]}&height={size[0]}&width={size[1]}&format=gtiff"

    request_hash = hashlib.md5(params.encode()).hexdigest()
    tiff_path = f"wcsCache/{request_hash}.tiff"

    if os.path.exists(tiff_path):
        image = imageio.imread(tiff_path)
        return np.array(image)
    
    else:
        response = requests.get(url_wcs + params)

        image = imageio.imread(response.content)
        imageio.imwrite(tiff_path, image)

        return np.array(image)


if __name__ == "__main__":
    #wms_url = "https://api.dataforsyningen.dk/dhm_DAF?service=WMS"
    wms_url = "https://api.dataforsyningen.dk/orto_foraar_DAF?service=WMS"
    TOKEN = "f8dbeb10b068e37b646751d5da8ffaaf"
    wms = WebMapService(wms_url, version="1.3.0", timeout=60)
    bbox = (725000, 6170000, 725100, 6170100)
    
    im_size = (
        8 * (bbox[2] - bbox[0]),
        8 * (bbox[3] - bbox[1]),
    )  # (width, height) approximately 8 pixels per meter as WMS resolution is 12.5 cm/pixel
    GISimage = get_bbox_dataforsyningen(bbox, wms, TOKEN, size=im_size)

    # WCS
    GISdem = get_bbox_dataforsyningen_wcs(bbox, TOKEN, layer="dhm_terraen", size=im_size)

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(GISimage)
    axes[0].set_title("WMS Image")
    axes[1].imshow(GISdem)
    axes[1].set_title("WCS Image")
    plt.show()

    
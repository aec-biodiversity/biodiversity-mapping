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

TOKEN = "f8dbeb10b068e37b646751d5da8ffaaf"
WMS_ENDPOINT = "https://api.dataforsyningen.dk/orto_foraar_DAF?service=WMS"


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

    images = []
    for layer in layers:
        # Calculate the hash for the request
        params_string = f"{wms.url}{wms.version}{bbox}{layer}{size}{transparent}"
        request_hash = hashlib.md5(params_string.encode()).hexdigest()
        png_path = f"./wmsCache/{request_hash}.png"
        
        if os.path.exists(png_path):
            image = Image.open(png_path)
            images.append(image)
        
        else:
            wms_response = wms.getmap(
                layers=[layer],
                srs="EPSG:25832",
                bbox=bbox,
                size=size,
                format="image/png",
                transparent=transparent,
                token=token,
            )
            image = Image.open(io.BytesIO(wms_response.read()))
            image.save(png_path)
            images.append(image)
    
    return images


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
    tiff_path = f"../wcsCache/{request_hash}.tiff"

    if os.path.exists(tiff_path):
        image = imageio.imread(tiff_path)
        return np.array(image)
    
    else:
        response = requests.get(url_wcs + params)

        image = imageio.imread(response.content)
        imageio.imwrite(tiff_path, image)

        return np.array(image)


def fetch_images(bbox, size, **kwargs):
    wms = WebMapService(WMS_ENDPOINT, version="1.3.0", timeout=60)
    images = get_bbox_dataforsyningen(bbox, wms, TOKEN, size=size, **kwargs)

    return images
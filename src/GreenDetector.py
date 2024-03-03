import numpy as np

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from PIL import Image
import plotly.graph_objects as go

import imgfetcher

## Green detection
def _reshape_img(img):
    """Return image as m-by-#channels matrix"""
    return np.reshape(img, newshape=(-1, img.shape[-1]), order="C")


def gis2np(img):
    """Return image as m-by-n-by-3 ndarray"""
    return np.array(img.convert("RGB"))


def rel_channel(img, channel: int = 0):
    """Return relative weight of a channel"""
    y = img[:,:,channel] / img.sum(axis=-1)
    return y


def SVC(img, y_train):
    """Return a binary support vector classifier"""
    X_train = _reshape_img(img)
    clf = make_pipeline(
            StandardScaler(),
            LinearSVC(dual="auto", random_state=0, tol=1e-5),
    )
    clf.fit(X_train, np.ravel(y_train))
    return clf


def predict_img(svc, img):
    """Return classification for SVC"""
    X = _reshape_img(img)
    y_hat = svc.predict(X)
    return np.reshape(y_hat, img.shape[:2])


def corr(img, tgt_vec=[1, 0.5, 0.5]):
    """Return correlation with target vector"""
    x_mag = np.linalg.norm(img, ord=2, axis=2, keepdims=True)
    b = np.asarray(tgt_vec)
    x = np.tensordot(img / x_mag, b, axes=1)
    return x


def fetch_green_mask(bbox, size, method="corr", **kwargs):
    
    # Method is tuned for this layer type
    layers = ["geodanmark_2023_12_5cm_cir"]
    gis_img = imgfetcher.fetch_img(bbox, size, layers=layers, **kwargs)[0]

    img = gis2np(gis_img)
    if method == "corr":
        x = corr(img)
        return x > 1.15
    elif method == "threshold":
        x = rel_channel(img)
        return x > 0.4
    else:
        raise NotImplementedError("Use method 'corr' or 'threshold'")

import laspy

def points_in_bbox(las_file, bbox):
    with laspy.open(las_file) as f:
        las = f.read()
    
    xyz = np.vstack((
        las.X * las.header.scale[0],
        las.Y * las.header.scale[1],
        las.Z * las.header.scale[2],
    ))

    xmin, xmax = bbox[0], bbox[2]
    ymin, ymax = bbox[1], bbox[3]
    in_box = (
        (xmin <= xyz[0,:]) & (xyz[0,:] <= xmax) &
        (ymin <= xyz[1,:]) & (xyz[1,:] <= ymax)
    )

    classes = las.classification[in_box]
    return xyz[:, in_box], classes

import matplotlib.pyplot as plt

COLORS = {
    5: [0.2, .6, .37],
    6: [.8, .3, .2],
    2: [0.5, .9, .3],
    4: [.6, .5, .3]
}

LAS_FILE = r"./data/PUNKTSKY_1km_6170_720.las"
def make_tree_plot(bbox, n_pts=60000):
    xyz, classes = points_in_bbox(LAS_FILE, bbox)

    colors = np.zeros((classes.size, 3))
    for k, v in COLORS.items():
        idx = classes == k
        colors[idx, :] = np.array(v)
    
    rng = np.random.default_rng()
    idx = rng.integers(0, xyz.shape[1], n_pts)

    fig = go.Figure(data=[go.Scatter3d(
        x=xyz[0, idx],
        y=xyz[1, idx],
        z=xyz[2, idx],
        mode='markers',
        marker=dict(color=colors[idx, :]),
    )])
    fig.update_traces(marker_size = 1)
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(aspectmode="cube"),
    )

    return fig


def make_green_plot(bbox):
    layers = ["geodanmark_2023_12_5cm_cir"]
    im_size = (
        8 * (bbox[2] - bbox[0]),
        8 * (bbox[3] - bbox[1]),
    )
    images = imgfetcher.fetch_images(bbox, im_size, layers=layers)
    img = [gis2np(img) for img in images][0]

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot()
    ax.imshow(img)

    green_mask = corr(img) > 1.15
    green_fraction = green_mask.sum() / green_mask.size

    return fig, green_fraction


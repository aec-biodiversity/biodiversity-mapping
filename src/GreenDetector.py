import numpy as np

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from PIL import Image

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
    gis_img = imgfetcher.fetch_img(bbox, size, layers=layers, **kwargs)

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

LAS_FILE = r"../data/PUNKTSKY_1km_6170_720.las"
def make_3d_plot(bbox, n_pts=10000):
    xyz, classes = points_in_bbox(LAS_FILE, bbox)

    rng = np.random.default_rng()
    idx = rng.integers(0, xyz.shape[1], n_pts)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(xyz[0, idx], xyz[1, idx], xyz[2, idx], c=classes[idx], s=1)
    return fig
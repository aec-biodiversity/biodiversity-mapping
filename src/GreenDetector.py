import numpy as np

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from PIL import Image

def _reshape_img(img):
    """Return image as m-by-3 matrix"""
    return np.reshape(img, newshape=(-1, 3), order="C")


def gis2np(img):
    """Return image as m-by-n-3 ndarray"""
    return np.array(img.convert("RGB"))


def rel_channel(img, channel: int = 0, threshold: float = 0.4):
    """Return relative weight of a channel"""
    y = img[:,:,channel] / img.sum(axis=-1)
    y[y < threshold] = 0
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
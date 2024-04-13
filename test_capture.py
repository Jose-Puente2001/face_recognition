import pytest
import tkinter as tk
import cv2


def test_face_recognition():
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    assert isinstance(faceCascade, cv2.CascadeClassifier)

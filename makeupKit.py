import numpy as np
import faceBlendCommon as fbc
import cv2
import dlib


def eyesLandMarks(readImg):
    # Load Image
    readImg = cv2.imread(r"images\img.png")
    imgRGB = cv2.cvtColor(readImg, cv2.COLOR_BGR2RGB)

    # Detect face landmarks
    PREDICTOR_PATH = r"shape_predictor_68_face_landmarks.dat"

    faceDetector = dlib.get_frontal_face_detector()
    landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)
    landmarks = fbc.getLandmarks(faceDetector, landmarkDetector, imgRGB)

    # Left Eye Center
    left_eye = ((np.asarray(landmarks[37]) + np.asarray(landmarks[38]) + np.asarray(landmarks[40]) + np.asarray(
        landmarks[41])) / 4).astype(int)
    # Right Eye Center
    right_eye = ((np.asarray(landmarks[43]) + np.asarray(landmarks[44]) + np.asarray(landmarks[46]) + np.asarray(
        landmarks[47])) / 4).astype(int)

    # Draw face landmarks
    for point in landmarks:
        cv2.circle(readImg, point, 1, (200, 0, 0), -1)

    # Draw Eye's centers landmarks
    cv2.circle(readImg, tuple(left_eye), 2, (0, 255, 255), -1)
    cv2.circle(readImg, tuple(right_eye), 2, (0, 255, 255), -1)
    readImg = cv2.cvtColor(readImg, cv2.COLOR_BGR2RGB)
    return imgRGB, readImg

import glob
import pickle

import cv2
import numpy as np

#==============================================================================
def findCorners(image, nx, ny):
    # Find the chessboard corners
    return cv2.findChessboardCorners(image, (nx, ny), None)

#------------------------------------------------------------------------------
def drawCorners(image, nx, ny, corners, ret):
    cv2.drawChessboardCorners(image, (nx, ny), corners, ret)

#------------------------------------------------------------------------------
def calibrate():
    images = glob.glob('camera_cal/calibration*.jpg')
    imageSize = (720, 1280)
    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane

    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) # x, y coordinates

    for fname in images:
        image = cv2.imread(fname)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = findCorners(gray, 9, 6)

        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imageSize, None, None)

    camera_pickle = {}
    camera_pickle["mtx"] = mtx
    camera_pickle["dist"] = dist
    pickle.dump(camera_pickle, open( "camera_cal/camera_cal.p", "wb"))

    return ret, mtx, dist, rvecs, tvecs

#==============================================================================
def getCameraCoefficients():
    '''
    Get pre-calibrated camera coefficients for undistorting images

    :return: dist array, and mtx array of camera coefficients
    '''
    # Get from pickled file
    with open('camera_cal/camera_cal.p', mode='rb') as f:
        camera_data = pickle.load(f)
    mtx = camera_data["mtx"]
    dist = camera_data["dist"]

    return dist, mtx
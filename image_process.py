import numpy as np
import cv2
import matplotlib.pyplot as plt

from moviepy.editor import VideoFileClip

# Camera coefficients precalculated form camera.py
# These only need to be calculated once per camera and can be stored as constants
MTX = np.asarray([[1.15396093e+03, 0.00000000e+00, 6.69705357e+02],
                  [0.00000000e+00, 1.14802496e+03, 3.85656234e+02],
                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
DIST = np.asarray([[-2.41017956e-01, -5.30721173e-02, -1.15810355e-03,
                    -1.28318856e-04, 2.67125290e-02]])

#==============================================================================
def readImage(imagePath):
    '''
    Read an image as an array using opencv

    :param imagePath: Path to image
    :return: image array in RGB color space.
    '''
    return cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)

def showImage(name, img):
    '''
    Displays an image in a new window.

    :param name: string name of the window
    :param img: image to display
    :return: None
    '''
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

#------------------------------------------------------------------------------
def grayscale(img, display=False):
    '''
    Convert BGR image to Grayscale

    :param image: 3 channel image array in RGB color space
    :return: single channel grayscale image
    '''
    temp_image = np.copy(img)
    gray = cv2.cvtColor(temp_image, cv2.COLOR_RGB2GRAY)
    if display:
        showImage('Grayscale Image', gray)
    return gray


#------------------------------------------------------------------------------
def convertToHLS(img, display=False):
    '''
    Convert BGR image to RGB

    :param image: 3 channel image array in RGB color space
    :return: 3 channel HLS image
    '''
    temp_image = np.copy(img)
    return cv2.cvtColor(temp_image, cv2.COLOR_RGB2HLS)

#------------------------------------------------------------------------------
def convertToHSV(img):
    '''
    Convert BGR image to HSV

    :param image: 3 channel image array in RGB color space
    :return: 3 channel HSV image
    '''
    temp_image = np.copy(img)
    return cv2.cvtColor(temp_image, cv2.COLOR_RGB2HSV)

#------------------------------------------------------------------------------
def convertToLAB(img):
    '''
    Convert BGR image to LAB

    :param image: 3 channel image array in RGB color space
    :return: 3 channel LAB image
    '''
    temp_image = np.copy(img)
    return cv2.cvtColor(temp_image, cv2.COLOR_RGB2LAB)

#------------------------------------------------------------------------------
def convertToLUV(img):
    '''
    Convert BGR image to LUV

    :param image: 3 channel image array in RGB color space
    :return: 3 channel LUV image
    '''
    temp_image = np.copy(img)
    return cv2.cvtColor(temp_image, cv2.COLOR_RGB2LUV)

#------------------------------------------------------------------------------
def convertToYCRCB(img):
    '''
    Convert BGR image to YCrCb

    :param image: 3 channel image array in RGB color space
    :return: 3 channel YCrCb image
    '''
    temp_image = np.copy(img)
    return cv2.cvtColor(temp_image, cv2.COLOR_RGB2YCR_CB)

#------------------------------------------------------------------------------
def convertToYUV(img):
    '''
    Convert BGR image to YUV

    :param image: 3 channel image array in RGB color space
    :return: 3 channel YUV image
    '''
    temp_image = np.copy(img)
    return cv2.cvtColor(temp_image, cv2.COLOR_RGB2YUV)

def undistort(image, mtx, dist, display=False):
    '''
    Undistort an image given pre-calibrated camera coefficients

    :param image: RGB Image
    :param mtx: camera matrix array
    :param dist: camera distortion coefficients
    :param display: bool whether to view the image in a new window.
    :return: ndistorted RGB image
    '''
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    if display:
        showImage('Undistorted Image', cv2.cvtColor(undist, cv2.COLOR_RGB2BGR))

    return undist

#------------------------------------------------------------------------------
def absSobelThreshold(img, orient='x', sobel_kernel=3, thresh=(0, 255), display=False):
    '''
    Creates a binary image using the sobel operator in either the x or y direction.
    This returns an image in the 0-1 range which may need to be scaled and cast
    back to uint8 to be properly displayed or written out to a file.

    :param img: Input RGB image
    :param orient: string 'x' or 'y' to specify the direction of the gradient.
    :param sobel_kernel: the size of the kernel to use
    :param thresh: tuple (low, high) representing the lower and upper bounds to threshold
    :return: binary image in 0-1 range
    '''
    temp_img = np.copy(img)

    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(temp_img, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(temp_img, cv2.CV_64F, 0, 1))

    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    if display:
        showImage('Sobel {0} Binary Threshold'.format(orient), (grad_binary*255).astype('uint8'))

    return grad_binary

def hlsBinary(image, channel, thresh=(0, 255), display=False):
    '''
    Create a binary image from the specified channel in the hls colorspace.

    :param image: Input RGB Image
    :param channel: Channel to extract from
    :param thresh: tuple containing thresh_min and thresh_max values
    :param display: book whether to display the output
    :return:
    '''
    hls = convertToHLS(image)
    ch = hls[:,:,channel].copy()
    hls_binary = np.zeros_like(ch)
    hls_binary[(ch >= thresh[0]) & (ch <= thresh[1])] = 1

    if display:
        showImage('HLS Binary Threshold', (hls_binary*255).astype('uint8'))

    return hls_binary

def hsvBinary(image, channel, thresh=(0, 255), display=False):
    '''
    Create a binary image from the specified channel in the hsv colorspace.

    :param image: Input RGB Image
    :param channel: Channel to extract from
    :param thresh: tuple containing thresh_min and thresh_max values
    :param display: book whether to display the output
    :return:
    '''
    hsv = convertToHSV(image)
    ch = hsv[:,:,channel].copy()
    hsv_binary = np.zeros_like(ch)
    hsv_binary[(ch >= thresh[0]) & (ch <= thresh[1])] = 1

    if display:
        showImage('HSV Binary Threshold', (hsv_binary*255).astype('uint8'))

    return hsv_binary

def labBinary(image, channel, thresh=(0, 255), display=False):
    '''
    Create a binary image from the specified channel in the lab colorspace.

    :param image: Input RGB Image
    :param channel: Channel to extract from
    :param thresh: tuple containing thresh_min and thresh_max values
    :param display: book whether to display the output
    :return: binary image of the specified channel and threshold
    '''
    lab = convertToLAB(image)
    ch = lab[:,:,channel].copy()
    lab_binary = np.zeros_like(ch)
    lab_binary[(ch >= thresh[0]) & (ch <= thresh[1])] = 1

    if display:
        showImage('LAB Binary Threshold', (lab_binary*255).astype('uint8'))

    return lab_binary

def luvBinary(image, channel, thresh=(0, 255), display=False):
    '''
    Create a binary image from the specified channel in the luv colorspace.

    :param image: Input RGB Image
    :param channel: Channel to extract from
    :param thresh: tuple containing thresh_min and thresh_max values
    :param display: book whether to display the output
    :return: binary image of the specified channel and threshold
    '''
    luv = convertToLUV(image)
    ch = luv[:,:,channel].copy()
    luv_binary = np.zeros_like(ch)
    luv_binary[(ch >= thresh[0]) & (ch <= thresh[1])] = 1

    if display:
        showImage('LUV Binary Threshold', (luv_binary*255).astype('uint8'))

    return luv_binary

def ycbcrBinary(image, channel, thresh=(0, 255), display=False):
    '''
    Create a binary image from the specified channel in the YCrCb colorspace.

    :param image: Input RGB Image
    :param channel: Channel to extract from
    :param thresh: tuple containing thresh_min and thresh_max values
    :param display: book whether to display the output
    :return: binary image of the specified channel and threshold
    '''
    ycrcb = convertToYCRCB(image)
    ch = ycrcb[:,:,channel].copy()
    ycrcb_binary = np.zeros_like(ch)
    ycrcb_binary[(ch >= thresh[0]) & (ch <= thresh[1])] = 1

    if display:
        showImage('LUV Binary Threshold', (ycrcb_binary*255).astype('uint8'))

    return ycrcb_binary

def yuvBinary(image, channel, thresh=(0, 255), display=False):
    '''
    Create a binary image from the specified channel in the yuv colorspace.

    :param image: Input RGB Image
    :param channel: Channel to extract from
    :param thresh: tuple containing thresh_min and thresh_max values
    :param display: book whether to display the output
    :return: binary image of the specified channel and threshold
    '''
    yuv = convertToYUV(image)
    ch = yuv[:,:,channel].copy()
    yuv_binary = np.zeros_like(ch)
    yuv_binary[(ch >= thresh[0]) & (ch <= thresh[1])] = 1

    if display:
        showImage('LUV Binary Threshold', (yuv_binary*255).astype('uint8'))

    return yuv_binary


#------------------------------------------------------------------------------
def source_points(image, points):
    '''
    Draws four reference points onto an image

    :param imagePath: path to the image to load
    :param points: list of 4 tuples with (x, y) coordinates for points to draw
    :return: image with points
    '''
    pt1, pt2, pt3, pt4 = points

    circles = cv2.circle(image.copy(), pt1, 10, (255, 0, 0), 5)
    circles = cv2.circle(circles, pt2, 10, (255, 0, 0), 5)
    circles = cv2.line(circles, pt1, pt2, 10, (255, 0, 0), 5)
    circles = cv2.circle(circles, pt3, 10, (255, 0, 0), 5)
    circles = cv2.line(circles, pt2, pt3, 10, (255, 0, 0), 5)
    circles = cv2.circle(circles, pt4, 10, (255, 0, 0), 5)
    circles = cv2.line(circles, pt3, pt4, 10, (255, 0, 0), 5)

    return circles

#------------------------------------------------------------------------------
# Initialize global variables to store x and y points for lane lines
last_leftx = None
last_lefty = None
last_rightx = None
last_righty = None
def pipeline(image, displayType='final', display=False):
    '''
    Image processing pipeline for rendering lane boundaries over an input image.

    :param image: Input image in RGB color space
    :return: Output image with lane boundary drawn over the top.
    '''
    global last_leftx
    global last_lefty
    global last_rightx
    global last_righty

    # Perform all image processing and image transformation to create a binary
    # perspective transformed image.
    Minv, combined, undist, warped_rgb, warped_thresh = processImage(image, display)

    # Get left and right lane pixel coordinates
    leftx, lefty, rightx, righty = getLaneLineCoordinates(warped_thresh, windows=10, width=50, height=720)

    # Fit a curve to the left and right lane lines
    left_fitx, right_fitx, left_curverad, right_curverad = fitCurve(leftx, lefty, rightx, righty, display=display)

    # Draw lines on image based on the fit curve
    result = drawLaneLines(image, undist, warped_thresh, left_fitx, lefty,
                           right_fitx, righty, Minv, left_curverad, right_curverad, display=display)

    # Display the result
    if display:
        showImage('Lane Lines Overlay', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        cv2.imwrite('output_images/final_result.png', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

    # Update last know lane points
    last_leftx = leftx
    last_lefty = lefty
    last_rightx = rightx
    last_righty = righty

    if displayType == 'undistorted':
        return undist
    elif displayType == 'threshold':
        return cv2.cvtColor(combined, cv2.COLOR_GRAY2RGB)
    elif displayType == 'warped_rgb':
        return warped_rgb
    elif displayType == 'warped_threshold':
        return cv2.cvtColor(warped_thresh, cv2.COLOR_GRAY2RGB)
    elif displayType == 'final':
        return result

#------------------------------------------------------------------------------
def processImage(image, display):
    '''
    Perform all image processing and transformation for the lane finding pipeline.

    :param image: Input image in RGB format
    :param display: bool whether to show the image process at each stage
    :return: processed images and inverse matrix for unwarping the perspective transform.
    '''

    # Undistort image
    undist = undistort(image, MTX, DIST, display=display)

    # Threshold image
    gray = grayscale(undist, display=display)
    sobelx_binary = absSobelThreshold(gray, orient='x', thresh=(20, 100),
                                      display=display)
    sobely_binary = absSobelThreshold(gray, orient='y', thresh=(20, 100),
                                      display=display)
    #s_binary = hsvBinary(undist, 1, thresh=(120, 255), display=display)
    b_binary = labBinary(undist, 2, thresh=(150, 255), display=display)
    ycbcr_binary = ycbcrBinary(undist, 0, thresh=(210, 255), display=display)

    # Combine binary thresholds
    combined = np.zeros_like(gray)
    combined[((sobelx_binary == 1) & (sobely_binary == 1)) | (b_binary == 1) | (ycbcr_binary == 1)] = 1
    combined = (combined.copy() * 255).astype('uint8')
    if display:
        showImage('Combined Binary Threshold', combined)
        cv2.imwrite('output_images/combined_binary.png', combined)

    # Choose source reference points for performing a perspective transform
    # TODO: Find a way to generate these dynamically
    src = np.float32([[516, 460], [756, 460], [1200, 720], [100, 720]])

    # Select points for undistorted image
    dst = np.float32([[0, 0], [1280, 0], [1050, 720], [250, 720]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Create perspective transformed image
    imageSize = (image.shape[1], image.shape[0])
    warped_rgb = cv2.warpPerspective(undist, M, imageSize)
    if display:
        showImage('Warped RGB Image', cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite('output_images/perspective_transform.png', cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2BGR))

    warped_thresh = cv2.warpPerspective(combined, M, imageSize)
    if display:
        showImage('Warped Threshold Image', warped_thresh)
        cv2.imwrite('output_images/perspective_threshold.png', warped_thresh)

    return Minv, combined, undist, warped_rgb, warped_thresh

#------------------------------------------------------------------------------
def getLaneLineCoordinates(warped_thresh, windows=10, width= 50, height=720):
    '''
    Get the x, y coordinates of the pixels corresponding to the left and
    right lane lines.

    :param windows:
    :param end:
    :param warped_thresh: perspective transformed binary threshold image
    :return: coordinates of lane pixels as 4 arrays leftx, lefty, rightx, righty
    '''
    global last_leftx
    global last_lefty
    global last_rightx
    global last_righty

    leftx = []
    lefty = []
    rightx = []
    righty = []
    end = height-1

    # Find peaks in lower half of thresholded image
    histogram = np.sum(warped_thresh[warped_thresh.shape[0] / 2:, :].copy(), axis=0)
    left_center = histogram[:histogram.shape[0] / 2].argmax()
    right_center = histogram[histogram.shape[0] / 2:].argmax() + \
                   histogram.shape[0] / 2

    # Use a sliding window to detect left and right lane pixels
    for window in range(windows):
        start = end - (height/windows - 1)
        left_points = np.transpose(warped_thresh[start:end,
                                   left_center - width:left_center + width].nonzero())
        right_points = np.transpose(warped_thresh[start:end,
                                    right_center - width:right_center + width].nonzero())
        # Add back the window offset
        left_points[:, 0] = left_points[:, 0].copy() + start
        left_points[:, 1] = left_points[:, 1].copy() + left_center - width + 1
        right_points[:, 0] = right_points[:, 0].copy() + start
        right_points[:, 1] = right_points[:, 1].copy() + right_center - width + 1

        # Store x and y values
        # [HACK] Bug with cv2.fillpoly() flipping portions of the mask
        # Have to reduce the number of points to get mask to display properly
        # http://stackoverflow.com/questions/37392128/wrong-result-using-function-fillpoly-in-opencv-for-very-large-images
        # Extract x and y values for the pixels that make up the left and right lanes
        if left_points.any():
            left_center = int(left_points[:, 1].copy().mean())
            # Place a point at the top edge of the image so lane extends
            # the full length of the perspective transform
            if start == 0:
                leftx.append(left_center)
                lefty.append(0)
            # Place a point at the bottom edge of the image so lane extends
            # the full length of the perspective transform
            elif end == 719:
                leftx.append(left_center)
                lefty.append(end)
            # For all other points use the mean for the center in the x direction
            # and use the end so that line is made of evenly spaced points.
            else:
                leftx.append(left_center)
                lefty.append(end)
        # Repeat the process with the right lane pixels
        if right_points.any():
            right_center = int(right_points[:, 1].copy().mean())
            if start == 0:
                rightx.append(right_center)
                righty.append(0)
            elif end == 719:
                rightx.append(right_center)
                righty.append(end)
            else:
                rightx.append(right_center)
                righty.append(end)
        end = start - 1

    # Convert from list to array
    leftx = np.asarray(leftx)
    lefty = np.asarray(lefty)
    rightx = np.asarray(rightx)
    righty = np.asarray(righty)

    # If no lane pixels were detected use the last known lane values
    # If no last known values are known use some default
    if not leftx.any():
        leftx = last_leftx
        lefty = last_lefty
    if not rightx.any():
        rightx = last_rightx
        righty = last_righty
    if leftx is None:
        leftx = np.array([0])
        lefty = np.array([0])
    if rightx is None:
        rightx = np.array([1279])
        righty = np.array([0])

    return leftx, lefty, rightx, righty

def fitCurve(leftx, lefty, rightx, righty, display=False):
    '''
    Fit a curve to the detected left and right lane pixels.

    :param leftx: x coordinates of the left lane line
    :param lefty: y coordinates of the left lane line
    :param rightx: x coordinates of the right lane line
    :param righty: y coordinates of the right lane line
    :return: fit curves for the left and right lane lines
    '''
    # Fit a line to the points
    # Fit a second order polynomial to each lane line
    left_fit = np.polyfit(lefty, leftx, 2)
    left_fitx = left_fit[0] * lefty ** 2 + left_fit[1] * lefty + left_fit[2]
    right_fit = np.polyfit(righty, rightx, 2)
    right_fitx = right_fit[0] * righty ** 2 + right_fit[1] * righty + \
                 right_fit[2]

    if display:
        # Plot the data
        plt.plot(leftx, lefty, 'o', color='red')
        plt.plot(rightx, righty, 'o', color='blue')
        plt.xlim(0, 1280)
        plt.ylim(0, 720)
        plt.plot(left_fitx, lefty, color='green', linewidth=3)
        plt.plot(right_fitx, righty, color='green', linewidth=3)
        plt.gca().invert_yaxis() # to visualize as we do the images
        plt.savefig('output_images/curve_fit.png')
        plt.clf()

    # Get Curvature of the Radius
    lefty_eval = np.max(lefty)
    left_curverad = ((1 + (
    2 * left_fit[0] * lefty_eval + left_fit[1]) ** 2) ** 1.5) \
                    / np.absolute(2 * left_fit[0])
    righty_eval = np.max(righty)
    right_curverad = ((1 + (
    2 * right_fit[0] * righty_eval + right_fit[1]) ** 2) ** 1.5) \
                     / np.absolute(2 * right_fit[0])

    # Convert to World Space From Pixel Space
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meteres per pixel in x dimension
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    left_curverad = ((1 + (
    2 * left_fit_cr[0] * lefty_eval + left_fit_cr[1]) ** 2) ** 1.5) \
                    / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (
    2 * right_fit_cr[0] * righty_eval + right_fit_cr[1]) ** 2) ** 1.5) \
                     / np.absolute(2 * right_fit_cr[0])

    return left_fitx, right_fitx, left_curverad, right_curverad

#------------------------------------------------------------------------------
def drawLaneLines(image, undist, warped_thresh, left_fitx, lefty, right_fitx,
                  righty, Minv, left_curverad, right_curverad, display=False):
    '''
    Draw lane lines over the original undistorted image.

    :param image:
    :param undist:
    :param warped_thresh:
    :param left_fitx:
    :param lefty:
    :param right_fitx:
    :param righty:
    :param Minv:
    :param display:
    :return:
    '''
    # Draw Lines
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped_thresh).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, lefty]))])
    pts_right = np.array(
            [np.flipud(np.transpose(np.vstack([right_fitx, righty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    mask = cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    if display:
        showImage('Lane Line Mask', mask)
        cv2.imwrite('output_images/lane_mask.png', mask)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv,
                                  (image.shape[1], image.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    # Calculate distance from center and overlay text
    xm_per_pix = 3.7 / 700
    lane_difference = pts_right[0, pts_right[0,:,1].argmax(), 0] - pts_left[0, pts_left[0,:,1].argmax(), 0]
    center_offset = (lane_difference - result.shape[1]/2) * xm_per_pix
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result,'Estimated Curvature Left: {:.4f}m Right: {:.4f}m'.format(left_curverad, right_curverad),(100,100), font, 1,(255,255,255),3,cv2.LINE_AA)
    cv2.putText(result, 'Distance from center: {:.2f}m'.format(center_offset),(100,140), font, 1,(255,255,255),3,cv2.LINE_AA)
    return result

#------------------------------------------------------------------------------
def processVideo(clip, type):
    '''
    Processed video images and returns the type of display specified.

    :param clip: input video clip
    :param type: type of video to display
    :return: processed video clip
    '''
    def process(image):
        return pipeline(image, displayType=type)
    return clip.fl_image(process)

#------------------------------------------------------------------------------
def extractVideoFrames():
    '''
    Uses moveipy and the pipeline developed above to render each frame of the
    input video as a given type.

    :return: output processed video
    '''
    videoType = 'final'
    project_output = 'video/{0}.mp4'.format(videoType)
    input_clip = VideoFileClip("project_video.mp4")
    #project_clip = input_clip.fl_image(pipeline)
    project_clip = input_clip.fx(processVideo, videoType)
    project_clip.write_videofile(project_output, audio=False)
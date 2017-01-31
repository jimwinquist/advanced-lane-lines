**Advanced Lane Finding Project**

[//]: # (Image References)

[image1]: ./camera_cal/calibration1.jpg "Original"
[image2]: ./output_images/undistorted/calibration1.jpg "Undistorted"
[image3]: ./test_images/test1.jpg "Road Original"
[image4]: ./output_images/undistorted/test1.jpg "Road Undistorted"
[image5]: ./output_images/sobel_thresholds/sobel_x_threshold.png "Sobel X Threshold"
[image6]: ./output_images/sobel_thresholds/sobel_y_threshold.png "Sobel Y Threshold"
[image7]: ./output_images/hls_thresholds/hls_threshold.png "HLS S Channel Threshold"
[image8]: ./output_images/lab_thresholds/lab_threshold.png "LAB B Channel Threshold"
[image9]: ./output_images/ycrcb_thresholds/ycrcb_threshold.png "YCrCb Y Channel Threshold"
[image10]: ./output_images/combined_thresholds/combined_threshold.png "Combined Threshold"
[image11]: ./output_images/source_points.png "Source points"
[image12]: ./output_images/warped_image.png "Warp Example"
[image13]: ./output_images/perspective_transform.png "Perspective Transform"
[image14]: ./output_images/perspective_threshold.png "Perspective Threshold"
[image15]: ./output_images/curve_fit.png "Fit Visual"
[image16]: ./output_images/lane_mask.png "Lane Mask"
[image17]: ./output_images/final_result.png "Output"
[video1]: ./video/final.mp4 "Video"

###Camera Calibration

I began by calibrating the camera and pre-calculating camera coefficients to use to undistort each image in the pipeline to remove the effects of camera distortion. The module camera.py and specifically the function calibrate() contains the code I used for calibrating the camera. 

Using a variety of chessboard images taken from different angles I was able to use opencv `findChessboardCorners()` to extract a large number of `objectpoints` which are the (x,y,z) coordinates of the chessboard corners in the world, and `imagepoints` which represent the (x,y) pixel coordinates of the corners in the image. These object points and image points are then passed to opencv's `cv2.calibrateCamera()` function to get back the camera matrix and distortion coefficients. Once these are calculated I wrote them out to a pickle file for further use because these coefficients can be passed to opencv `cv2.undistort()` to undistort any image taken from this camera in the future. 

## Undistorting the images
| Original        | Undistorted     |
| :-------------: | :-------------: |
|![Original][image1] | ![Undistorted][image2]|

##Pipeline (single images)

Below is an example of how the pre-calculated camera matrix and distortion coefficients are applied to an image taken from a front facing camera in the car. By using `cv2.undistort()` the road images can be undistorted just like the chessboard calibration images.

| Original        | Undistorted     |
| :-------------: | :-------------: |
|![Original][image3] | ![Undistorted][image4]|

### Color Transformation and Thresholding

A significant portion of this project was spent on trying to find the best color spaces and thresholding parameters to choose to help isolate just the lane line pixels. I experimented with converting the original camera images to a variety of color spaces including rgb, hls, hsv, luv, lab, ycbcr, and yuv. I also experimented with using adaptive thresholding, and sobel gradient thresholding, as well as a combination of all of the above to find the most effective binary image to use as a base for isolating lane pixels to determine the curvature of the road. Several examples of different color space thresholds can be seen in the images below.

![Sobel X Threshold][image5]
![Sobel Y Threshold][image6]
![HLS Threshold][image7]
![LAB Threshold][image8]
![YCrCb Threshold][image9]

It is important to find a threshold that will handle road surface color variation, as well as changing lighting conditions. Many thresholds are able to handle gradients well or find the yellow lines very well, or the white lines. However, it is only by combining multiple gradients together, that it is possible to get the best of all possibilities. 

My code for image processing is in the `image_process.py` module in the function called `processImage()` and the thresholding process begins at line 370. In the end the threshold that I found to work the best was a combination of using sobel x and y gradients mixed with a thresholded b channel form the Lab color space, and the Y channel from the YCbCr color space.

![Combined Threshold][image10]

###Perspective Transform

The code for my perspective transform is also in the `image_process.py` module on line 391 of the `processImage()` function. To perform the perspective transform I experimented with different source points that lie on a plain, and through trial and error came up with a fixed set of points that works well for warping the image. Using fixed points worked relatively well for this project, since the car in the example video isn't going up or down any really steep inclines. Because of this we can assume that there will always be a similar fixe plane projected into the image out in from of the car.

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 516, 460      | 0, 0          | 
| 756, 460      | 1280, 0       |
| 1200, 720     | 1050, 720     |
| 100, 720      | 250, 720      |

I checked that the source points made sense in the context of the original image, and also verified that for a test image with straight lines that the destination points in the perspective transformed image were also straight and parallel. 

| Source Points   | Perspective Transform   |
| :-------------: | :---------------------: |
|![Source Points][image11] | ![Warped Image][image12]|


I calculated the transformation matrix M using opencv's `cv2.getPerspective()` and then used `cv2.warpPerspective()` to transform the undistorted image into a birds-eye-view perspective transform. Both the warped image and warped threshold image can be seen below, to show how the transformed threshold image can be used to fit a polynomial to the curved lane lines.

| Perspective Transform  | Perspective Threshold   |
| :--------------------: | :---------------------: |
|![Perspective Transform][image13] | ![Perspective Threshold][image14]|


###Detect Lane Lines and Fit a Polynomial

I implemented a sliding window to detect the lane pixels from the perspectived transformed threshold image. My code for finding the lane lines is in the `image_process.py` module in a function called `getLaneLineCoordinates()`, and it starts by taking a histogram of the threshold image to find the two largest peaks. It then proceeds to step up the image in discrete windows and pull out the coordinates of all the lane pixels. Once I accurately detect all of the lane pixels and separate them into the right and left lane line, I was able to use the detected coordinates to fit a polynomial to each lane. I did this in the function `fitCurve()` which takes x and y coordinates for the left and right lanes as input, and then returns the fit line and the curvature of the radius. 

| Curve Fit            | Lane Mask            |
| :------------------: | :------------------: |
|![Curve Fit][image15] | ![Lane Mask][image16]|


###Curvature of Radius

The curvature of the radius must first be calculated in pixel space(x, y) coordinates, and then converted back into meters based on the image dimensions to match what the car is seeing in the real world. And the cars position with respect to center can be calculated by taking the difference between the lane center and the center of the image.

I did this in lines 540 through 560 in my code in `image_process.py` in the `fitCurve()` function.


###Draw Lane Lines

I implemented this step in lines 565 through 603 in my code in `image_process.py` in the function `drawLaneLines()`.  In addition to adding the fitted lane lines back to the original image, I calculated the offset of the car from the center of the lane and added text to display both this offset and the curvature of radius for both the right and left lane lines. Here is an example of my result on a test image:

![Final Output][image17]

---

###Pipeline (video)
After combining all the pieces of the pipeline I was able to accurately draw the lane back onto the original undistorted image, and successfully track the lane lines on every frame without much deviation from the actual lane. The process for handling video used the same pipeline as for images. Each frame in the video is passed to `pipeline()` in the `image_process.py` module, and the image is processed according to the process detailed above, and then converted back into a video clip.

Here's a [link to my video result](./video/final.mp4)

---

###Discussion

One of the key points to making this pipeline work were the experimentation with various types of thresholding across a large range of different color channels. Without the ability to accurately identify pixels that belong to the lane lines it is very hard to perform any of the subsequent processing. Much of the process relies on having a good binary threshold that can separate the lane lines from the rest of the background image. While I was able to create a working example, I think a lot more study would need to be done on how to detect lane lines in a much wider range of lighting conditions, and with a variety of road surfaces to account for color variation. Since the video displayed here was taken on a clear day in ideal weather conditions, it would be necessary to test the pipeline out in a much larger variety of situations. For further research I would like to shoot additional video of different weather conditions, extreme road glare, driving in fog, snow, daytime vs. nightime driving, light road surfaces vs. dark road surfaces, poorly marked lane lines, etc. In addition, I would also like to work to create a pipeline that isn't based only on hard coded but that adapts to a changing environment. It would be exciting to explore deriving the source points to do the perspective transform dynamically based on the horizon, or detecting lanes to the left and right for lane changing, or varying the threshold used based on detected lighting conditions. 

Overall, I am happy with this first pass of the lane finding pipeline and am excited to keep exploring.


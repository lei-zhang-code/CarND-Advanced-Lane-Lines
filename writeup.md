## Advanced Lane Line Project Writeup

---

**Goals and steps**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[chess_undistort]: ./output_images/chessboard_undistortion.png "Undistorted chessboard image."
[test_image]: ./output_images/test_image_undistort.png "Original test image."
[color_grad_thresh]: ./output_images/color_grad_threshold.png "Color and gradient thresholds."
[trapezoid]: ./output_images/trapezoid.png "Trapezoid"
[perspective_transform]: ./output_images/perspective_transform.png "Perspective transform"
[lane_line_bootstrap]: ./output_images/lane_line_bootstrap.png "Lane line bootstrap."
[lazy_search]: ./output_images/lazy_search.png "Lane line search from the previous polynomial curve."
[final_image]: ./output_images/final_image.png "Lane line overlayed to original image."
[video1]: ./output_videos/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the Camera Calibration section in "advanced_lane_lines.ipynb".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][chess_undistort]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][test_image]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image in the 'Finding a good threshold for S' and 'Directional gradient thresholds' section.  Here's an example of my output for this step.

![alt text][color_grad_thresh]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The perspective transform is done through a mapping between trapezoid on a straight-line image and a rectangle, as shown below:

![alt_text][trapezoid]

 I first find the perspective transform matrix M via

```python
tpzd = [[580, 460], [700, 460], [1105, 719], [205, 719]]
rect = [[500, 0], [2000, 0], [2000, 1400], [500, 1400]]
M = cv2.getPerspectiveTransform(np.float32(tpzd), np.float32(rect))
```

I apply the perspective transform via

```python
warped = cv2.warpPerspective(undistorted, M, (2500, 1400), flags=cv2.INTER_LINEAR)
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][perspective_transform]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

For the very first frame, I first find out the starting position of the two lanes at the bottom of the image by using the histogram at the lower half of the image space. Then from the base, I applied 9 searching windows with a margin of 150 pixel to find the lane line points from bottom to top as shown in the following figure:

![alt text][lane_line_bootstrap]

To extract the polynomial fitting of the lane lines, I used a method different from the class. I first compare left and right side to see which side has more points. I then fit the side of more points with a 2nd order polynomial `x = A * y^2 + B * y + C`.


In order to make sure the left and right lanes have the same curvature, the side of fewer points is assumed to have the same A and B coefficients. We only fit for C in the fewer point side.

```python
# Fit a second order polynomial to each
if len(leftx) > len(rightx):
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = [left_fit[0], left_fit[1], compute_lane_offset_px(rightx, righty, left_fit[0], left_fit[1])]
else:
    right_fit = np.polyfit(righty, rightx, 2)
    left_fit = [right_fit[0], right_fit[1], compute_lane_offset_px(leftx, lefty, right_fit[0], right_fit[1])]
```
where `compute_lane_offset_px()` is define as:
```python
def compute_lane_offset_px(x, y, A, B):
    return np.mean(x - A * y**2 - B * y)
```

For all the rest of the frames, I find the lane line pixels without using the sliding window approach as I used for the first frame. Instead, I used the polynomial extracted from the previous frame plus a margin to search for lane line points:

I used a queue structure to save the past 5 frames' left and right fittings. When I draw the polynomial line, I used the averaged polynomial line of all 5 frames.

![alt_text][lazy_search]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Once I have the left and right line polynomials `left_fix` and `right_fit`, I computed the curvature as below:

```python
def compute_curvature(left_fit, right_fit, y_eval=1400):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 3/200 # Each long dash is 3 meters, it occupies about 200 pixels in the warped image.
    xm_per_pix = 3.7/1500 # Lane distance is 3 meter, occupying 1500 pixels in the warped image.
    # Calculate the new radii of curvature
    left_A = left_fit[0] * xm_per_pix / (ym_per_pix**2)
    left_B = left_fit[1] * xm_per_pix / ym_per_pix
    left_curverad = ((1 + (2*left_A*y_eval*ym_per_pix + left_B)**2)**1.5) / np.absolute(2*left_A)
    right_A = right_fit[0] * xm_per_pix / (ym_per_pix**2)
    right_B = right_fit[1] * xm_per_pix / ym_per_pix
    right_curverad = ((1 + (2*right_A*y_eval*ym_per_pix + right_B)**2)**1.5) / np.absolute(2*right_A)
    # Now our radius of curvature is in meters
    return (left_curverad + right_curverad) / 2.0
```

The offset from the center of the lane is computed as:

```python
def compute_off_center(left_fit, right_fit, y_eval=1400, center_pix=1225):
    left_x_bottom = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_x_bottom = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    mid_x = (left_x_bottom + right_x_bottom) / 2.0
    off_center_px = center_pix - mid_x
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    return off_center_px * xm_per_pix
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I overlayed the detected lanes back to the original image as shown below:

![alt text][final_image]

The right lane is not as good as the left lane. This is because the left lane does not have as many points as the right lane in the warped birdview image.

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The polynomial fitting does not match with the original video very well at far distance. This is because the data at far end has more uncertainty, but there are more of them in the warped bird-eye image. Whereas the near end of the lanes has much better data points and accuracy, but the appear very short in the bird-eye image.


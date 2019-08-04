from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
from imutils import contours

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required = True,
# 	help = "Path to the image to be scanned")
# args = vars(ap.parse_args())



# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread("img/answered-sheet-photo.jpg")
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)
 
# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)
 
# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
 
	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break
 
# apply the four point transform to obtain a top-down
# view of the original image
paper = four_point_transform(image, screenCnt.reshape(4, 2))
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)


warped =imutils.resize(warped , height=650)

#identify circles ..........
warped = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
warped = cv2.GaussianBlur(warped, (5, 5), 0)
# Read image
 # Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
 
# # Change thresholds
# params.minThreshold = 10
# params.maxThreshold = 200
 
# Filter by Area.
params.filterByArea = True
params.minArea = 100
 
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.8
 
# # Filter by Convexity
# params.filterByConvexity = True
# params.minConvexity = 0.87
 
# # Filter by Inertia
# params.filterByInertia = True
# params.minInertiaRatio = 0.01
 
# Create a detector with the parameters
# ver = (cv2.__version__).split('.')
# if int(ver[0]) < 3 :
#     detector = cv2.SimpleBlobDetector(params)
# else : 
detector = cv2.SimpleBlobDetector_create(params)
# # Set up the detector with default parameters.
# detector = cv2.SimpleBlobDetector_create()
 
# # Detect blobs.
keypoints = detector.detect(warped)
# x = keypoints[i].pt[0] #i is the index of the blob you want to get the position
# y = keypoints[i].pt[1]


# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(warped, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# # Show keypoints
# for keyPoint in keypoints:
#     x = keyPoint.pt[0]
#     y = keyPoint.pt[1]
#     s = keyPoint.size
# key_arr=[]
# for i in keypoints[0].pt:
# 	# x=keypoints[i].pt[0]
# 	# y=keypoints[i].pt[1]
	
# 	print (i)
# np.float([kp[idx].pt for idx in range(0, len(kp))]).reshape(-1, 1, 2)
# pts = ([keypoints[idx].pt for idx in range(0, len(keypoints))])

# for i in pts:
	
# 	print (i)

pts= cv2.KeyPoint_convert(keypoints)
# pts = pts.reshape(-1, 1, 2)
# print(pts)
# print(keypoints[0].pt)

print(pts)
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)

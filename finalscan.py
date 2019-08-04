# import the necessary packages
from transform import four_point_transform
from skimage.filters import threshold_local
from skimage import measure
import numpy as np
import argparse
import cv2
import imutils
from imutils import contours

 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())



# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread(args["image"])
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
# warped = threshold_local(warped, 11, offset = 10, method = "gaussian")
# warped = cv2.threshold(warped, 0, 255,
 	# cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[0]

#hough circles
# circles = cv2.HoughCircles(warped,cv2.HOUGH_GRADIENT,1,20,
#                             param1=50,param2=30,minRadius=10,maxRadius=20)

# circles = np.uint16(np.around(circles))
# thresh = cv2.threshold(warped, 0, 255,
# 		cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# # mask = cv2.bitwise_and(thresh, thresh)
# # total = cv2.countNonZero(mask)

# print (circles)
# font = cv2.FONT_HERSHEY_SIMPLEX
# height, width = warped.shape[:2] 
# for i in circles[0,:]:
    
#     # draw the outer circle
#     cv2.circle(warped,(i[0],i[1]),i[2],(0,255,0),2)
#     # draw the center of the circle
#     cv2.circle(warped,(i[0],i[1]),2,(0,0,255),3)
#     cv2.putText(warped,'btn' + str(i),(i[0]+10,i[1]+i[2]+10), font, 0.5, (200,255,155), 1, cv2.LINE_AA)

# # for i in circles[0,:]:
# #     # draw the outer circle
# #     cv2.circle(warped,(i[0],i[1]),i[2],(0,255,0),2)
# #     # draw the center of the circle
# #     cv2.circle(warped,(i[0],i[1]),2,(0,0,255),3)

# cv2.imshow('detected circles',warped)

# # hough_test67

# circles = cv2.HoughCircles(warped,cv2.HOUGH_GRADIENT,1,20,
#                             param1=50,param2=30,minRadius=10,maxRadius=20)

# circles = np.uint16(np.around(circles))
# thresh = cv2.threshold(warped, 0, 255,
# 		cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# # thresh = cv2.erode(thresh, None, iterations=2)
# # thresh = cv2.dilate(thresh, None, iterations=4)
# # mask = cv2.bitwise_and(thresh, thresh)
# # total = cv2.countNonZero(mask)

# # print (circles)
# font = cv2.FONT_HERSHEY_SIMPLEX
# height, width = warped.shape[:2] 
# labels= measure.label(thresh, neighbors=8, background=0)
# mask = np.zeros(thresh.shape, dtype="uint8")
# for i in circles[0,:]:
#     circle_ma= np.zeros(thresh.shape,dtype="uint8")
#     # circle_ma[circles== i]=255
# 	# draw the outer circle
#     cv2.circle(mask,(i[0],i[1]),i[2],(0,255,0),2)
#     # draw the center of the circle
#     cv2.circle(mask,(i[0],i[1]),2,(0,0,255),3)
#     # mask = cv2.bitwise_and(thresh, thresh, mask=mask)
#     total = cv2.countNonZero(circle_ma)
#     if total>300:
#         mask= cv2.add(mask, circle_ma)
# 		# print(circles)
# 		# cv2.circle(warped,(i[0],i[1]),i[2],(0,255,0),2)
# 		# cv2.putText(warped,'btn' + str(i),(i[0]+10,i[1]+i[2]+10), font, 0.5, (200,255,155), 1, cv2.LINE_AA)


# # for i in circles[0,:]:
# #     # draw the outer circle
# #     cv2.circle(warped,(i[0],i[1]),i[2],(0,255,0),2)
# #     # draw the center of the circle
# #     cv2.circle(warped,(i[0],i[1]),2,(0,0,255),3)



# cv2.imshow('detected circles',warped)
# cv2.imshow('mask',mask)


# blob detection

detector = cv2.SimpleBlobDetector_create()
 
# Detect blobs.
keypoints = detector.detect(warped)
 
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(warped, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# Show keypoints
print(keypoints)
cv2.imshow("Keypoints", im_with_keypoints)

# show the original and scanned images
print("STEP 3: Apply perspective transform")
# cv2.imshow("preprocessed", warped)
cv2.imshow("Original", imutils.resize(orig, height = 650))
# cv2.imshow("Scanned", imutils.resize(warped, height = 650))
# cv2.imshow("Thresh", imutils.resize(thresh, height = 650))

cv2.imshow("preprocessed", warped)
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)

cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.drawContours(image, [screenCnt], -1, color, 2)
# cv2.drawContours(image, [warped], -1, (0, 255, 0), 2)
# import the necessary packages
from transform import four_point_transform
from skimage.filters import threshold_local
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

circles = cv2.HoughCircles(warped,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=10,maxRadius=20)

circles = np.uint16(np.around(circles))
thresh = cv2.threshold(warped, 0, 255,
		cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# mask = cv2.bitwise_and(thresh, thresh)
# total = cv2.countNonZero(mask)

print (circles)
font = cv2.FONT_HERSHEY_SIMPLEX
height, width = warped.shape[:2] 
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(warped,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(warped,(i[0],i[1]),2,(0,0,255),3)
    cv2.putText(warped,'btn' + str(i),(i[0]+10,i[1]+i[2]+10), font, 0.5, (200,255,155), 1, cv2.LINE_AA)

# for i in circles[0,:]:
#     # draw the outer circle
#     cv2.circle(warped,(i[0],i[1]),i[2],(0,255,0),2)
#     # draw the center of the circle
#     cv2.circle(warped,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',warped)

# try to do the same for only the shaded regions

# for i in circles[0,:]:
# 	mask = np.zeros(thresh.shape, dtype="uint8")
# 	mask = cv2.bitwise_and(thresh, thresh, mask=mask)
# 	total = cv2.countNonZero(mask)
# 	if total>None:
# 		cv2.circle(warped, (i[0], i[1],i[2], (0,255,0)),2)
# 		cv2.circle(warped, (i[0],i[1],2,(0,0,255),3)
	
		
# cv2.imshow('detected circles',warped))



#end identify circles ..........


# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect

# warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
# T = threshold_local(warped, 11, offset = 10, method = "gaussian")
# warped = (warped > T).astype("uint8") * 255

# image_color= cv2.imread("img/answered-sheet-photo.jpg")

# image_ori = cv2.cvtColor(image_color,cv2.COLOR_BGR2GRAY)

#new code ...........
# warped =(imutils.resize(warped, height=650))
# image_ori = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)

# lower_bound = np.array([0,0,10])
# upper_bound = np.array([255,255,195])


# img= warped
# mask = cv2.inRange(warped, lower_bound, upper_bound)

# mask = cv2.adaptiveThreshold(image_ori,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#             cv2.THRESH_BINARY_INV,33,2)

# kernel = np.ones((3, 3), np.uint8)

# #Use erosion and dilation combination to eliminate false positives. 
# #In this case the text Q0X could be identified as circles but it is not.
# mask = cv2.erode(mask, kernel, iterations=6)
# mask = cv2.dilate(mask, kernel, iterations=3)

# closing = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
#         cv2.CHAIN_APPROX_SIMPLE)[0]
# contours.sort(key=lambda x:cv2.boundingRect(x)[0])

# array = []
# ii = 1
# print(len(contours))
# for c in contours:
#     (x,y),r = cv2.minEnclosingCircle(c)
#     center = (int(x),int(y))
#     r = int(r)
#     if r >= 6 and r<=10:
#         cv2.circle(img,center,r,(0,255,0),2)
#         array.append(center)

# # cv2.imshow("preprocessed", warped)
 #new code ends ..............




# warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
# warped = cv2.GaussianBlur(warped, (5, 5), 0)
# # thresh = threshold_local(warped, 11, offset = 10, method = "gaussian")
# thresh = cv2.threshold(warped, 0, 255,
# 	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# # warped = (warped > T).astype("uint8") * 255
 
# cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
# 	cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# questionCnts = []

# for c in cnts:
# 	# compute the bounding box of the contour, then use the
# 	# bounding box to derive the aspect ratio

# 	(x, y, w, h) = cv2.boundingRect(c)
# 	ar = w / float(h)
 
# 	# in order to label the contour as a question, region
# 	# should be sufficiently wide, sufficiently tall, and
# 	# have an aspect ratio approximately equal to 1
# 	if w >= 10 and h >= 10 and ar >= 0.6 and ar <= 1.1:
# 		questionCnts.append(c)

# # sort the question contours top-to-bottom, then initialize
# # the total number of correct answers
# questionCnts = contours.sort_contours(questionCnts,
# 	method="top-to-bottom")[0]
# correct = 0
 
# # each question has 5 possible answers, to loop over the
# # question in batches of 5
# for (q, i) in enumerate(np.arange(0, len(questionCnts), 2)):
# 	# sort the contours for the current question from
# 	# left to right, then initialize the index of the
# 	# bubbled answer
# 	cnts = contours.sort_contours(questionCnts[i:i + 5])[0]
# 	bubbled = None

# 	# loop over the sorted contours
# 	for (j, c) in enumerate(cnts):
# 		# construct a mask that reveals only the current
# 		# "bubble" for the question
# 		mask = np.zeros(thresh.shape, dtype="uint8")
# 		cv2.drawContours(mask, [c], -1, 255, -1)
 
# 		# apply the mask to the thresholded image, then
# 		# count the number of non-zero pixels in the
# 		# bubble area
# 		mask = cv2.bitwise_and(thresh, thresh, mask=mask)
# 		total = cv2.countNonZero(mask)
 
# 		# if the current total has a larger number of total
# 		# non-zero pixels, then we are examining the currently
# 		# bubbled-in answer
# 		if bubbled is None or total > bubbled:
# 			bubbled = (total, j)

# 	# initialize the contour color and the index of the
# 	# *correct* answer
# 	color = (0, 0, 255)
# 	k = ANSWER_KEY[q]
 
# 	# check to see if the bubbled answer is correct
# 	if k == bubbled[1]:
# 		color = (0, 255, 0)
# 		correct += 1
 
# 	# draw the outline of the correct answer on the test
# 	# cv2.drawContours(paper, [cnts[k]], -1, color, 3)


# grab the test taker
# score = (correct / 5.0) * 100
# print("[INFO] score: {:.2f}%".format(score))
# cv2.putText(paper, "{:.2f}%".format(score), (10, 30),
# 	cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
# cv2.imshow("Original", image)

# cv2.imshow("Exam", paper)

# show the original and scanned images
print("STEP 3: Apply perspective transform")
# cv2.imshow("preprocessed", warped)
cv2.imshow("Original", imutils.resize(orig, height = 650))
# cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.imshow("Thresh", imutils.resize(thresh, height = 650))

cv2.imshow("preprocessed", warped)
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)

cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.drawContours(image, [screenCnt], -1, color, 2)
# cv2.drawContours(image, [warped], -1, (0, 255, 0), 2)
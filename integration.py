# import the necessary packages
from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
from imutils import contours
import argparse
import cv2
import math
import numpy as np


def scanner(image):
	image =cv2.imread("img/answered-sheet-photo.jpg")
# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
	image = cv2.imread("image")
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

# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect

# warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
# T = threshold_local(warped, 11, offset = 10, method = "gaussian")
# warped = (warped > T).astype("uint8") * 255

	warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	warped = cv2.GaussianBlur(warped, (5, 5), 0)
# thresh = threshold_local(warped, 11, offset = 10, method = "gaussian")
	thresh = cv2.threshold(warped, 0, 255,
		cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# warped = (warped > T).astype("uint8") * 255
 
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	questionCnts = []

	for c in cnts:
	# compute the bounding box of the contour, then use the
	# bounding box to derive the aspect ratio

		(x, y, w, h) = cv2.boundingRect(c)
		ar = w / float(h)
 
	# in order to label the contour as a question, region
	# should be sufficiently wide, sufficiently tall, and
	# have an aspect ratio approximately equal to 1
		if w >= 10 and h >= 10 and ar >= 0.6 and ar <= 1.1:
			questionCnts.append(c)

# sort the question contours top-to-bottom, then initialize
# the total number of correct answers
	questionCnts = contours.sort_contours(questionCnts,
		method="top-to-bottom")[0]
	correct = 0
 
# each question has 5 possible answers, to loop over the
# question in batches of 5
	for (q, i) in enumerate(np.arange(0, len(questionCnts), 2)):
	# sort the contours for the current question from
	# left to right, then initialize the index of the
	# bubbled answer
		cnts = contours.sort_contours(questionCnts[i:i + 5])[0]
		bubbled = None

	# loop over the sorted contours
		for (j, c) in enumerate(cnts):
		# construct a mask that reveals only the current
		# "bubble" for the question
			mask = np.zeros(thresh.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
 
		# apply the mask to the thresholded image, then
		# count the number of non-zero pixels in the
		# bubble area
			mask = cv2.bitwise_and(thresh, thresh, mask=mask)
			total = cv2.countNonZero(mask)
 
		# if the current total has a larger number of total
		# non-zero pixels, then we are examining the currently
		# bubbled-in answer
			if bubbled is None or total > bubbled:
				bubbled = (total, j)

	# initialize the contour color and the index of the
	# *correct* answer
		color = (0, 0, 255)
		k = ANSWER_KEY[q]
 
	# check to see if the bubbled answer is correct
		if k == bubbled[1]:
			color = (0, 255, 0)
			correct += 1
 
	# draw the outline of the correct answer on the test
	# cv2.drawContours(paper, [cnts[k]], -1, color, 3)


# grab the test taker
# score = (correct / 5.0) * 100
# print("[INFO] score: {:.2f}%".format(score))
# cv2.putText(paper, "{:.2f}%".format(score), (10, 30),
# 	cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
# cv2.imshow("Original", image)

# cv2.imshow("Exam", paper)

# show the original and scanned images
	# print("STEP 3: Apply perspective transform")
	# cv2.imshow("Original", imutils.resize(orig, height = 650))
	# cv2.imshow("Scanned", imutils.resize(warped, height = 650))
	# cv2.imshow("Thresh", imutils.resize(thresh, height = 650))

	# cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)

	# cv2.imshow("Outline", image)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	return cv2.imshow("Scanned", imutils.resize(warped, height = 650))

CORNER_FEATS = (
    0.322965313273202,
    0.19188334690998524,
    1.1514327482234812,
    0.998754685666376,
)

TRANSF_SIZE = 512


def normalize(im):
    return cv2.normalize(im, np.zeros(im.shape), 0, 255, norm_type=cv2.NORM_MINMAX)

def get_approx_contour(contour, tol=.01):
    """Get rid of 'useless' points in the contour"""
    epsilon = tol * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)

def get_contours(image_gray):
    im2, contours, heirarachy = cv2.findContours(
        image_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return map(get_approx_contour, contours)

def get_corners(contours):
    return sorted(
        contours,
        key=lambda c: features_distance(CORNER_FEATS, get_features(c)))[:4]

def get_bounding_rect(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return np.int0(box)

def get_convex_hull(contour):
    return cv2.convexHull(contour)

def get_contour_area_by_hull_area(contour):
    return (cv2.contourArea(contour) /
            cv2.contourArea(get_convex_hull(contour)))

def get_contour_area_by_bounding_box_area(contour):
    return (cv2.contourArea(contour) /
            cv2.contourArea(get_bounding_rect(contour)))

def get_contour_perim_by_hull_perim(contour):
    return (cv2.arcLength(contour, True) /
            cv2.arcLength(get_convex_hull(contour), True))

def get_contour_perim_by_bounding_box_perim(contour):
    return (cv2.arcLength(contour, True) /
            cv2.arcLength(get_bounding_rect(contour), True))

def get_features(contour):
    try:
        return (
            get_contour_area_by_hull_area(contour),
            get_contour_area_by_bounding_box_area(contour),
            get_contour_perim_by_hull_perim(contour),
            get_contour_perim_by_bounding_box_perim(contour),
        )
    except ZeroDivisionError:
        return 4*[np.inf]

def features_distance(f1, f2):
    return np.linalg.norm(np.array(f1) - np.array(f2))

# Default mutable arguments should be harmless here
def draw_point(point, img, radius=5, color=(0, 0, 255)):
    cv2.circle(img, tuple(point), radius, color, -1)

def get_centroid(contour):
    m = cv2.moments(contour)
    x = int(m["m10"] / m["m00"])
    y = int(m["m01"] / m["m00"])
    return (x, y)

def order_points(points):
    """Order points counter-clockwise-ly."""
    origin = np.mean(points, axis=0)

    def positive_angle(p):
        x, y = p - origin
        ang = np.arctan2(y, x)
        return 2 * np.pi + ang if ang < 0 else ang

    return sorted(points, key=positive_angle)

def get_outmost_points(contours):
    all_points = np.concatenate(contours)
    return get_bounding_rect(all_points)

def perspective_transform(img, points):
    """Transform img so that points are the new corners"""

    source = np.array(
        points,
        dtype="float32")

    dest = np.array([
        [TRANSF_SIZE, TRANSF_SIZE],
        [0, TRANSF_SIZE],
        [0, 0],
        [TRANSF_SIZE, 0]],
        dtype="float32")

    img_dest = img.copy()

    transf = cv2.getPerspectiveTransform(source, dest)
    warped = cv2.warpPerspective(img, transf, (TRANSF_SIZE, TRANSF_SIZE))
    return warped

def sheet_coord_to_transf_coord(x, y):
    return list(map(lambda n: int(np.round(n)), (
        TRANSF_SIZE * x/744.055,
        TRANSF_SIZE * (1 - y/1052.362)
    )))

def get_question_patch(transf, q_number):
    # Top left
    tl = sheet_coord_to_transf_coord(
        
        200,
        850 - 80 * (q_number - 1)
    )

    # Bottom right
    br = sheet_coord_to_transf_coord(
        650,
        800 - 80 * (q_number - 1)
    )
    return transf[tl[1]:br[1], tl[0]:br[0]]

def get_question_patches(transf):
    for i in range(1, 11):
        yield get_question_patch(transf, i)

def get_alternative_patches(question_patch):
    for i in range(5):
        x0, _ = sheet_coord_to_transf_coord(100 * i, 0)
        x1, _ = sheet_coord_to_transf_coord(50 + 100 * i, 0)
        yield question_patch[:, x0:x1]

def draw_marked_alternative(question_patch, index):
    cx, cy = sheet_coord_to_transf_coord(
        50 * (2 * index + .5),
        50/2)
    draw_point((cx, TRANSF_SIZE - cy), question_patch, radius=10, color=(255, 255, 0))

def get_marked_alternative(alternative_patches):
    means = list(map(np.mean, alternative_patches))
    sorted_means = sorted(means)

    # Simple heuristic
    if sorted_means[0]/sorted_means[1] > .7:
        return None

    return np.argmin(means)

def get_letter(alt_index):
    return ["A", "B", "Male", "D", "E"][alt_index] if alt_index is not None else "N/A"

def get_answers(source_file):
    """Run the full pipeline:

        - Load image
        - Convert to grayscale
        - Apply threshold
        - Find contours
        - Find corners among all contours
        - Find 'outmost' points of all corners
        - Apply perpsective transform to get a bird's eye view
        - Scan each line for the marked answer
    """

    # im_orig = cv2.imread(source_file)

    # blurred = cv2.GaussianBlur(im_orig, (11, 11), 10)

    # im = normalize(cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY))

    # ret, im = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)

    # contours = get_contours(im)
    # corners = get_corners(contours)

    # cv2.drawContours(im_orig, corners, -1, (0, 255, 0), 3)

    # outmost = order_points(get_outmost_points(corners))

    # transf = perspective_transform(im_orig, outmost)
    image = cv2.imread(source_file)
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

# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect

# warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
# T = threshold_local(warped, 11, offset = 10, method = "gaussian")
# warped = (warped > T).astype("uint8") * 255

    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped = cv2.GaussianBlur(warped, (5, 5), 0)
# thresh = threshold_local(warped, 11, offset = 10, method = "gaussian")
    thresh = cv2.threshold(warped, 0, 255,
		cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
    # warped= cv2.warpPerspective(img, transf, (TRANSF_SIZE, TRANSF_SIZE))
    # warped = imutils.resize(warped, height = 650)

    answers = []
    for i, q_patch in enumerate(get_question_patches(warped)):
        alt_index = get_marked_alternative(get_alternative_patches(q_patch))

        if alt_index is not None:
            draw_marked_alternative(q_patch, alt_index)

        answers.append(get_letter(alt_index))

    #cv2.imshow('orig', im_orig)
    #cv2.imshow('blurred', blurred)
    #cv2.imshow('bw', im)

    return answers, warped
    

# construct the argument parser and parse the arguments

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        help="Input image filename",
        required=True,
        type=str)

    parser.add_argument(
        "--output",
        help="Output image filename",
        type=str)

    parser.add_argument(
        "--show",
        action="store_true",
        help="Displays annotated image")

    args = parser.parse_args()

    answers, im = get_answers(args.input)

    for i, answer in enumerate(answers):
        print("Q{}: {}".format(i + 1, answer))

    if args.output:
        cv2.imwrite(args.output, im)
        print("Wrote image to {}".format(args.output))
        

    if args.show:
        cv2.imshow('trans', imutils.resize(im, height=500))

        print("Close image window and hit ^C to quit.")
        while True:
            cv2.waitKey()

if __name__ == '__main__':
    main()
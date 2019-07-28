from integration import get_answers
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


answers, img = get_answers("img/answered-sheet-photo.jpg")


cv2.imshow("Scanned", imutils.resize(img, height = 350))

for i, answer in enumerate(answers):
    print("Gender{}: {}".format(i + 1, answer))
    

# if args.output:
#     cv2.imwrite(args.output, im)
#     print("Wrote image to {}".format(args.output))
    
cv2.waitKey(0)
cv2.destroyAllWindows()


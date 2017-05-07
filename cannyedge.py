import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('tmp.png', 0)
ret, otsu = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
vis = img.copy()
# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()

# find and draw the keypoints
kp = fast.detect(img,None)
img2 = cv2.drawKeypoints(otsu, kp, None,color=(255,0,0))

# Print all default params
print("Threshold: ", fast.getThreshold())
print("nonmaxSuppression: ", fast.getNonmaxSuppression())
print("neighborhood: ", fast.getType())
print("Total Keypoints with nonmaxSuppression: ", len(kp))

cv2.imwrite('fast_true.png',img2)

# Disable nonmaxSuppression
fast.setNonmaxSuppression(0)
kp = fast.detect(img,None)

print ("Total Keypoints without nonmaxSuppression: ", len(kp))

img3 = cv2.drawKeypoints(otsu, kp, None, color=(255,0,0))

cv2.imwrite('fast_false.png',img3)




cv2.waitKey(0)
cv2.destroyAllWindows()

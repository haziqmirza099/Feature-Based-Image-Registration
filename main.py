import cv2
import numpy as np
from matplotlib import pyplot as plt


image1_path = ""  # add the image you want to resize or tranform
image2_path = ""  # add the reference name


image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)


gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)


orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(gray_image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray_image2, None)


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

matches = sorted(matches, key = lambda x:x.distance)

matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

matrix, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)

height, width = image2.shape[:2]
aligned_image = cv2.warpPerspective(image1, matrix, (width, height))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB))
plt.title("Aligned Image 1")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
plt.title("Image 2 (Reference)")
plt.axis("off")

plt.tight_layout()
plt.show()



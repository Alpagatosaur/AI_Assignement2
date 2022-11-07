import cv2
import os

doc_img = "img"
name_new_img = "template"
name_test_img = "test"

path_file = os.getcwd()
os.path.dirname(os.path.abspath(path_file))

PATH_TEST_IMG = os.path.join(path_file, doc_img, name_test_img + ".jpg")
PATH_NEW_IMG = os.path.join(path_file, doc_img, name_new_img  + ".jpg")


# Read the img
img = cv2.imread(PATH_TEST_IMG)

#Convert to gray
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Show img
cv2.imshow("Test image", img)

# After testing, the treshold value is setting to 100
vtresh = 100

ret, img_tresh = cv2.threshold(img_gray, vtresh, 255, cv2.THRESH_BINARY)

"""plt.figure(1)
plt.imshow(img_tresh, cmap = "gray")
plt.show()"""

# Search contours
contours, hierarchy = cv2.findContours(img_tresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

n = 0 # The n-th contours
k = 0 # the n-th contours' value
for i in range(len(contours)):
    if len(contours[i]) > k:
        n = i
        k = len(contours[i])
# cv2.drawContours(img_gray, contours, n, (255,255,255), 2)

# Get the template img
x, y, w, h = cv2.boundingRect(contours[n])
img_new = img[y:y+h, x:x+w]


# cv2.imshow("NEW", img_new)
# Save the template img
cv2.imwrite(PATH_NEW_IMG, img_new)

# Data for SIFT
color_new = cv2.cvtColor(img_new, cv2.IMREAD_GRAYSCALE)
color_test = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)


""" SIFT """
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(color_new, None)
kp2, des2 = sift.detectAndCompute(color_test, None)

bf = cv2.BFMatcher()

matches = bf.knnMatch(des1,des2,k=2)

good = []
for m,n in matches:

    if m.distance < 0.75*n.distance: # If the match value > 75%
        good.append([m])

fin = cv2.drawMatchesKnn(color_new,kp1,color_test,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("FINAL", fin)

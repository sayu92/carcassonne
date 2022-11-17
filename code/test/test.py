import cv2
import numpy as np

CANNY_LOW = 150
CANNY_HIGH = 200

HOUGH_RHO_STEP = 1
HOUGH_THETA_STEP = np.pi / 180
HOUGH_THRESHOLD = 150
HOUGH_MIN_LINE_LEN = 80
HOUGH_MAX_LINE_GAP = 30


img_rgb = cv2.imread("/home/louis-nicolas/carcassonne/code/test/carcassonne.jpg")

img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.blur(img_gray, (2, 2))
cv2.imshow("blur", img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# img = cv2.adaptiveThreshold(
#     img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 199, 5
# )
# cv2.imshow("Image_thre", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img = cv2.blur(img, (3, 3))
# cv2.imshow("blur", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

edges = cv2.Canny(img, CANNY_LOW, CANNY_HIGH)
cv2.imshow("canny", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

lines = cv2.HoughLinesP(
    edges,
    rho=HOUGH_RHO_STEP,
    theta=HOUGH_THETA_STEP,
    threshold=HOUGH_THRESHOLD,
    minLineLength=HOUGH_MIN_LINE_LEN,
    maxLineGap=HOUGH_MAX_LINE_GAP,
)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 3)

cv2.imshow("canny", img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()

threshold = 0.8
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
cv2.imshow("img", img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()

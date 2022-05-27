import cv2
import easyocr
import numpy as np

reader = easyocr.Reader(['ru'])

imgOriginal = cv2.imread("auto.jpg")
imgGray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)

ret, imgThreshold = cv2.threshold(imgGray, 100, 255, cv2.THRESH_BINARY)
# imgFiltered = cv2.bilateralFilter(imgThreshold, 11, 15, 15)
#imgCanny = cv2.Canny(imgOriginal, 100, 255)

contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

imgContours = np.zeros(imgOriginal.shape)
cv2.drawContours(imgContours, contours, -1, (0, 0, 255), 1)

foundNumbers = []

for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    area = cv2.contourArea(c)
    if (len(approx) >= 4) and (5000 <= area <= 10000):
        x, y, w, h = cv2.boundingRect(c)
        delta = w / h
        if 4.3 <= delta <= 5.3:
            foundNumbers.append([x, y, x + w, y + h])
            cv2.rectangle(imgOriginal, (x, y), (x + w, y + h), (255, 0, 0), 2)


counter = 0
for fn in foundNumbers:
    roi = imgThreshold[fn[1]:fn[3], fn[0]:fn[2]]
    name = "roi" + str(counter)
    cv2.imshow(name, roi)
    counter = counter + 1
    result = reader.readtext(roi)[0][1]
    cv2.putText(imgOriginal, result, (fn[0], fn[1] - 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    print(result)

cv2.imshow("imgOriginal", imgOriginal)

cv2.waitKey(0)

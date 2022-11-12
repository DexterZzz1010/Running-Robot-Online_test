def detect(readimg):
    gray = cv.cvtColor(readimg, cv.COLOR_BGR2GRAY)
    gray = cv.bilateralFilter(gray, 21, 75, 75)
    gray = cv.medianBlur(gray, 3)

    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    sobelx = cv.convertScaleAbs(sobelx)
    sobely = cv.convertScaleAbs(sobely)
    sobelxy = cv.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    edges = cv.Canny(sobelxy, 50, 150)


    circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, 1, 3000, param1=100, param2=30, minRadius=0, maxRadius=200)

    if np.any(circles != None):
        circles = np.uint16(np.around(circles))  # 取整
    else:
        circles = np.array([[[0, 0, 0]]])

    choose = circles[0, :]

    r, x, y = (choose[0, 2]), (choose[0, 0]), (choose[0, 1])
    return r, x, y



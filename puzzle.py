from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2

def findPuzzle(image,debug = False):
    
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,(7,7),0)
    
    thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    thresh = cv2.bitwise_not(thresh)
    
    contours = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    
    contours = sorted(contours,key=cv2.contourArea,reverse=True)
    
    puzzleCnt = None
    
    for c in contours:
        
        peri = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,0.02*peri,True)
        
        if len(approx) == 4:
            puzzleCnt = approx
            break
    
    if puzzleCnt is None:
        raise Exception(("Could not find Sudoku puzzle outline."))
    
    puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
    warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))
    
    return(puzzle,warped)

def extract_digit(cell):
    
    thresh = cv2.threshold(cell,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)
    
    contours = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    
    if len(contours) == 0:
        return None
    c = max(contours, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)
    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)
    
    if percentFilled < 0.03:
        return None
    
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)
    
    return digit
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
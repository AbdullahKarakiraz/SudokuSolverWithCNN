from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sudoku import Sudoku
from puzzle import findPuzzle
from puzzle import extract_digit
import numpy as np
import cv2

model = load_model("digit_classifier8.h5")

image = cv2.imread(r"C:\Users\Asus\Documents\SpyderScripts\SudokuProject\sudoku_image.jpeg")

puzzle_img,warped = findPuzzle(image)

stepX = warped.shape[1] // 9
stepY = warped.shape[0] // 9

board = np.zeros((9,9),dtype=int)
cellLocs = []

for y in range(0,9):
    row = []
    for x in range(0,9):
        
        startX = x * stepX
        startY = y * stepY
        endX = (x+1) * stepX
        endY = (y+1) * stepY
        
        cell = warped[startY:endY,startX:endX]
        digit = extract_digit(cell)
        if digit is not None:
            row.append((1,startX, startY, endX, endY))
            roi = cv2.resize(digit,dsize = (28,28))
            roi = roi.astype(float) / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis = 0)
            pred = model.predict(roi).argmax(axis=1)[0]
            board[y, x] = pred
        else:
            row.append((0,startX, startY, endX, endY))
    cellLocs.append(row)

    
puzzle = Sudoku(3,3,board=board.tolist())
solution = puzzle.solve()

for (cellRow,boardRow) in zip(cellLocs,solution.board):
    for (box,digit) in zip(cellRow,boardRow):
        
        ret,startX,startY,endX,endY = box
        if ret == 0:
            textX = int((endX - startX) * 0.33)
            textY = int((endY - startY) * -0.2)
            textX += startX
            textY += endY
            cv2.putText(puzzle_img, str(digit), (textX, textY),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
cv2.imshow("Result",puzzle_img)
            
            
            
            
            
            
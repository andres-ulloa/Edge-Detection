
import numpy as np
import cv2 as cv


#1 for vertical and 0 for horizontal
def sobelFiltering(img, direction):
    pass


#1 for vertical and 0 for horizontal
def laplaceFiltering(img, direction):
    pass


#1 for vertical and 0 for horizontal
def edgeDetection(img, direction, threshold):
    pass


def main():
    img = cv.imread('lena-grey.png')
    cv.imwrite(sobelFiltering(img, 0), 'sobel-x.png')
    
if __name__ == '__main__':
    main()
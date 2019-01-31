
import numpy as np
import cv2 as cv


def convolution(image, kernel):

    filtered_image = image
    num_rows = image.shape[1]
    num_cols = image.shape[0]

    kernel_size = (kernel.shape[0], kernel.shape[1])

    #asumes the kernel is simmetric and of odd dimensions
    edge = int(math.floor(kernel.shape[0]/2))

    for i in range(edge, num_rows - edge):
        for j in range(edge, num_cols - edge):
            window = fill_window(i,j, kernel_size, image)
            image[i,j] = get_linear_combination(window, kernel)
            

    return filtered_image


def fill_window(kernel_mid_x, kernel_mid_y, kernel_size, image):
    
    kernel = np.zeros((kernel_size[0], kernel_size[1]), dtype = int)
    from_x = kernel_mid_x - int(math.floor(kernel_size[1]/2))
    to_x = kernel_mid_x + int(math.floor(kernel_size[1]/2))
    from_y = kernel_mid_y -  int(math.floor(kernel_size[0]/2))
    to_y = kernel_mid_y +  int(math.floor(kernel_size[0]/2))

    for i in range(from_x, to_x + 1):
        for j in range(from_y, to_y + 1):

            kernel[i - from_x, j - from_y] = image[i,j]
            
    return kernel


def get_linear_combination(matrixA, matrixB):
    pass

#1 for vertical and 0 for horizontal
def sobelFiltering(img, direction):
    sobel_filter_x = np.matrix('-1,0,1;-2,0,2;-1,0,1')
    sobel_filter_y = np.matrix('-1,-2,-1;0,0,0;1,2,1')


def laplacianFiltering(img):
    laplacian_filter_x =  np.matrix('0,-1,0;-1,4,-1;0,-1,0')
    


#1 for vertical and 0 for horizontal
def edgeDetection(img, direction, threshold):
    pass


def main():
    img = cv.imread('lena-grey.png')
    cv.imwrite(sobelFiltering(img, 0), 'sobel-x.png')

if __name__ == '__main__':
    main()
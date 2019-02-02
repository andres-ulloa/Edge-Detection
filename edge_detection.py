
import numpy as np
import cv2 as cv
import math

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
    
    rows = matrixA.shape[0]
    cols = matrixA.shape[1]
    linear_comb = 0

    for i in range(0, rows):
        for j in range(0, cols):
            linear_comb += matrixA[i,j] * matrixB[i,j]

    return linear_comb


#1 for horizontal and 0 for vertical
def sobelFiltering(img, direction):

    sobel_filter_x = np.matrix('-1,0,1;-2,0,2;-1,0,1')
    sobel_filter_y = np.matrix('-1,-2,-1;0,0,0;1,2,1')

    gradients = np.zeros((img.shape[0], img.shape[1]), dtype = float)

    if direction == 0:

        gradients = convolution(img, sobel_filter_x)
    
    elif direction == 1:

        gradients = convolution(img, sobel_filter_y)

    else:
        print('nop')
        exit(0)
    
    return gradients


#1 for horizontal and 0 for vertical
def prewittFiltering(img, direction):
    prewitt_filter_x = np.matrix('-1,0,1;-1,0,1;-1,0,1')
    prewitt_filter_y = np.matrix('-1,-1,-1;0,0,0;1,1,1')

    gradients = np.zeros((img.shape[0], img.shape[1]), dtype = float)

    if direction == 0:
    
        gradients = convolution(img, prewitt_filter_x)
    
    elif direction == 1:

        gradients = convolution(img, prewitt_filter_y)

    else:
        print('nop')
        exit(0)

    return gradients     


def laplacianFiltering(img):
    laplacian_filter_x =  np.matrix('0,-1,0;-1,4,-1;0,-1,0')
    

def compute_edge_magnitude(gradients_x, gradients_y):
    gradient_magnitude = np.zeros((gradients_x.shape[0], gradients_x.shape[1]), dtype = float)
    for i in range(0, gradients_x.shape[0]):
        for j in range(0, gradients_y.shape[1]):
            gradient_magnitude[i,j] = abs(gradients_x[i,j]) + abs(gradients_y[i,j]) 

    return gradient_magnitude


def compute_edge_direction(gradients_x, gradients_y):
    gradient_direction = np.zeros((gradients_x.shape[0], gradients_x.shape[1]), dtype = float)
    for i in range(0, gradients_x.shape[0]):
        for j in range(0, gradients_y.shape[1]):
            gradient_direction[i,j] = math.atan(gradients_x[i,j] / gradients_y[i,j])

    return gradient_direction

def umbralize(img, thresh):

    umbralized_img = img

    for i in range(0 , img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i,j] < thresh:
                umbralized_img[i,j] = 0
            else:
                umbralized_img[i,j] = 255


#1 for vertical and 0 for horizontal
def edgeDetection(img, type_, threshold = 150):

    gradients_x =  np.zeros((img.shape[0], img.shape[1]), dtype = float)
    gradients_y =  np.zeros((img.shape[0], img.shape[1]), dtype = float)

    if type_ == 'sobel':
     
        gradients_x = sobelFiltering(img, 0)
        gradients_y = sobelFiltering(img, 1)
       
    elif type_ == 'prewitt':

        gradients_x = prewittFiltering(img, 0)
        gradients_y = prewittFiltering(img, 1)

    else:
        print('nop')
        exit(0)

    cv.imwrite(gradients_x, 'edge_detection_x.png')
    cv.imwrite(gradients_y, 'edge_detection_y.png')

    edge_magnitude = compute_edge_magnitude(gradients_x, gradients_y)
    edge_orientation = compute_edge_direction(gradients_x, gradients_y)
    cv.imwrite(edge_magnitude, 'pre_umbralized_mag.png')

    umbralized_mag = umbralize(edge_magnitude, threshold)
    cv.imwrite(umbralized_mag, 'umbralized_mag.png')
    


def main():

    img = cv.imread('lena-grey.png')
    cv.imwrite(sobelFiltering(img, 0), 'sobel-x.png')
    cv.imwrite(sobelFiltering(img,1),  'sobel-y.png')


if __name__ == '__main__':
    main()
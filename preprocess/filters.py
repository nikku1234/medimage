from PIL import Image 
from pylab import *
from matplotlib import pyplot as plt
from PIL import ImageFilter
from skimage.filters import threshold_otsu, threshold_adaptive
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage import io
from skimage import filters
from skimage import color
from scipy import ndimage as ndi
from skimage.morphology import watershed, disk
from skimage import data
from skimage.io import imread
from skimage.filters import rank
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte

'''
    Gaussian blur
    What this means is that when we create a kernel that follows a Gaussian distribution, the center
    pixel gets the most weight and its neighboring pixels get lesser weight when performing
    convolution. The pixel which has to be modified will have the highest weight in the kernel and
    the weight decreases for the pixels which are far away.
'''
def GaussianBlur(filename,value=5):
	img = Image.open(filename)
	blur_img = img.filter(ImageFilter.GaussianBlur(value))
	return blur_img

'''
    Median filter
    This is a very simple filter that returns the median value from the pixel and its neighbors.
'''

def MedianFilter(filename,value=7):
    img = Image.open(filename)
    blur_img_median = img.filter(ImageFilter.MedianFilter(value))
    return blur_img_median

'''
    Thresholding in image processing means to update the color value of a pixel to either white
    or black according to a threshold value. If the pixel value is greater than the threshold
    value, then set the pixel to WHITE, otherwise set it to BLACK. There are variations to
    thresholding as well. One of them is inverse thresholding, where we flip greater than to
    lesser than and everything else remains the same.
    '''
def threshold(filename):
    img = imread('nikhil.jpg')
    img = rgb2gray(img)
    thresh_value = threshold_otsu(img)
    thresh_img = img > thresh_value
    return thresh_img

'''
    Edge detection
    Sobel edge detector
    '''
def sobel_edge(filename):
    img = io.imread(filename)
    img = color.rgb2gray(img)
    edge = filters.sobel(img)
    return edge

'''
    Canny edge detector
    The Canny edge detector is another very important algorithm. It also uses the concept of
    gradients like in the Sobel edge detector, but in Sobel we only considered the magnitude of
    the gradient. In this we will also use the direction of the gradient to find the edges.
    This algorithm has four major steps:
    
    1. Smoothing: In this step, the Gaussian filter is applied to the image to reduce the noise in the image.
    
    2. Finding the gradient: After removing the noise, the next step is to find the gradient magnitude and direction by calculating the x-derivative and y-derivative. The direction is important, as the gradient is always perpendicular to the edge. Therefore, if we know the direction of the gradient, we can find the direction of the edges as well.
    
    3. Nonmaximal suppression: In this step, we check whether the gradient calculated is the maximum among the neighboring points lying in the positive and negative direction of the gradient; that is, whether it is the local maxima in the direction of the gradient. If it is not the local maxima, then that point is not part of an edge.
    
    4. Thresholding: In this algorithm, we use two threshold values--the high threshold and low threshold, unlike in Sobel where we just used one threshold value. This is called hysteresis thresholding. Let's understand how this works. We select all the edge points, which are above the high threshold and then we see if there are neighbors of these points which are below the high threshold but above the low threshold; then these neighbors will also be part of that edge. But if all the points of an edge are below the high threshold, then these points will not be selected.
    '''
def canny_edge(filename,value):
    img = io.imread(filename)
    img = color.rgb2gray(img)
    edge = feature.canny(img,value)
    return edge



def watershed(filename):
#img = data.astronaut()
    img = imread('nikhil.jpg')
    img_gray = rgb2gray(img)
    image = img_as_ubyte(img_gray)
#Calculate the local gradients of the image
#and only select the points that have a
#gradient value of less than 20
    markers = rank.gradient(image, disk(5)) < 20
    markers = ndi.label(markers)[0]
    gradient = rank.gradient(image, disk(2))
#Watershed Algorithm
    labels = watershed(gradient, markers)
    return labels


## for the GLCM
def GLCM(img):
    # Compute gray level cooccurence matrix and compute the features from it
    # Function returns the computed features
    properties = ['energy', 'homogeneity','dissimilarity','correlation','ASM']
    glcm = greycomatrix(img, [5], [0,45,90,135], levels=256, symmetric=True, normed=True)
    feats = np.hstack([greycoprops(glcm, prop).ravel() for prop in properties])
    return feats



## ADJUST DATA FOR MODEL INCLUDING IMAGES AND MASK
def adjustData(img,mask):
    img = img / 255
    mask = mask /255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return (img,mask)

This contains a python called the filters.py.This includes all the filters for the pre-processing of the images.

The contains 

	- GaussianBlur(filename,value=5) 
		returns the image after the GaussianBlur is applied

	- MedianFilter(filename,value=7)
		returns the image after the MedianFilter is applied

	- threshold(filename)
		returns the image after the threshold is applied with the help of otsu value

	- sobel_edge(filename)
		returns the image after the sobel_edge is applied

	- canny_edge(filename,value)
		returns the image after the canny_edge is applied

 	- watershed(filename):
 		returns the image after the canny_edge is ap

 	- GLCM(img)
 		return the features 

 	- adjustData(img,mask)
 		used in unet model for image and its mask, returns the image and mask

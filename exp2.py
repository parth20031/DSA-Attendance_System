import cv2

img=cv2.imread("sample.jpg")
height,width,_=img.shape
print(height)
print(width)
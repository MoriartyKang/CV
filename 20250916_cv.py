import cv2 as cv
import numpy as np
import sys

img=cv.imread('img\soccer.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

## 첫번째 예제
##cv.imshow('Title',img)
##cv.imshow('Upper left half',img[0:img.shape[0]//2,0:img.shape[1]//2,:])
##cv.imshow('Center',img[img.shape[0]//4:3*img.shape[0]//4,img.shape[1]//4:3*img.shape[1]//4])

## 두번쨰 예제
##t,bin_img=cv.threshold(img[:,:,2],0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
##print('오휴 알고리즘 최적 임계값=',t)

##cv.imshow('R channel',img[:,:,2])
##cv.imshow('R channel binarization',bin_img)

## 세번째 예쩨
##img=cv.resize(img,dsize=(0,0),fx=0.25,fy=0.25)

##def gamma(f,gamma=1.0):
##    f1=f/255.0
##    return np.uint8(255*(f1**gamma))

##gc=np.hstack((gamma(img,0.5),gamma(img,0.75),gamma(img,1.0),gamma(img,2.0),gamma(img,3.0)))
##cv.imshow('gamma',gc)

cv.waitKey()
cv.destroyAllWindows()
import cv2 as cv
import numpy as np

img=cv.imread('img\soccer.jpg')
##gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

## 소벨 연산자 에지
##grad_x=cv.Sobel(gray,cv.CV_32F,1,0,ksize=3)
##grad_y=cv.Sobel(gray,cv.CV_32F,0,1,ksize=3)

##sobel_x=cv.convertScaleAbs(grad_x)
##sobel_y=cv.convertScaleAbs(grad_y)

##edge_strength=cv.addWeighted(sobel_x,0.5,sobel_y,0.5,0)

##cv.imshow('Original',gray)
##cv.imshow('sobelx',sobel_x)
##cv.imshow('sobely',sobel_y)
##cv.imshow('edge strength',edge_strength)


## 캐니 에지
##canny1=cv.Canny(gray,50,150)
##canny2=cv.Canny(gray,100,200)

##cv.imshow('Orginal',gray)
##cv.imshow('Canny1',canny1)
##cv.imshow('Canny2',canny2)


## 캐니 에지 맵 경계선
##canny=cv.Canny(gray,100,200)

##contour,hierarchy=cv.findContours(canny,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)

##lcontour=[]
##for i in range(len(contour)):
##    if contour[i].shape[0]>100:
##        lcontour.append(contour[i])

##cv.drawContours(img,lcontour,-1,(0,255,0),3)

##cv.imshow('Original with contours',img)
##cv.imshow('Canny',canny)


## GrabCut 물체 분할
img_show = np.copy(img)

mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
mask[:, :] = cv.GC_PR_BGD

BrushSiz=9
LColor,RColor=(255,0,0),(0,0,255)

def painting(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img_show, (x, y), BrushSiz, LColor, -1)
        cv.circle(mask, (x, y), BrushSiz, cv.GC_FGD, -1)
    elif event == cv.EVENT_RBUTTONDOWN:
        cv.circle(img_show, (x, y), BrushSiz, RColor, -1)
        cv.circle(mask, (x, y), BrushSiz, cv.GC_BGD, -1)
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
        cv.circle(img_show, (x, y), BrushSiz, LColor, -1)
        cv.circle(mask, (x, y), BrushSiz, cv.GC_FGD, -1)
    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_RBUTTON:
        cv.circle(img_show, (x, y), BrushSiz, RColor, -1)
        cv.circle(mask, (x, y), BrushSiz, cv.GC_BGD, -1)
    cv.imshow('Painting', img_show)

cv.namedWindow('Painting')
cv.setMouseCallback('Painting', painting)

while(True):
    if cv.waitKey(1)==ord('q'):
        break

background=np.zeros((1,65),np.float64)
foreground=np.zeros((1,65),np.float64)

cv.grabCut(img, mask, None, background, foreground, 5, cv.GC_INIT_WITH_MASK)
mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')
grab = img * mask2[:, :, np.newaxis]

cv.imshow('Grab cut image',grab)

cv.waitKey()
cv.destroyAllWindows()
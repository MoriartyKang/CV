from PyQt5.QtWidgets import *
import sys
import winsound
import cv2 as cv 
import numpy as np

'''class BeepSound(QMainWindow):
    def __init__(self) :
        super().__init__()
        self.setWindowTitle('삑 소리 내기') ## 윈도우 이름, 위치 지정
        self.setGeometry(200,200,500,100)

        shortBeepButton=QPushButton('짧게 삑',self)	## 버튼 생성
        longBeepButton=QPushButton('길게 삑',self)
        quitButton=QPushButton('종료',self)
        self.label=QLabel('어서오시게!',self)
        
        shortBeepButton.setGeometry(10,10,100,30) ## 버튼 위치, 크기 지정
        longBeepButton.setGeometry(110,10,100,30)
        quitButton.setGeometry(210,10,100,30)
        self.label.setGeometry(10,40,500,70)
        
        shortBeepButton.clicked.connect(self.shortBeepFunction) ## 콜백 함수 지정
        longBeepButton.clicked.connect(self.longBeepFunction)         
        quitButton.clicked.connect(self.quitFunction)
       
    def shortBeepFunction(self):
        self.label.setText('주파수 1000 / 0.5초 / 삑 소리')   
        winsound.Beep(1000,500)
        
    def longBeepFunction(self):
        self.label.setText('주파수 1000 / 3초 / 삑 소리')        
        winsound.Beep(1000,3000) 
                
    def quitFunction(self):
        self.close()
                
app=QApplication(sys.argv) 
win=BeepSound() 
win.show()
app.exec_()'''

class Grabcut(QMainWindow):
    def __init__(self) :
        super().__init__()
        self.setWindowTitle('오림')
        self.setGeometry(200,200,700,200)
       
        fileButton=QPushButton('파일',self)
        paintButton=QPushButton('페인팅',self)
        cutButton=QPushButton('오림',self)
        incButton=QPushButton('+',self)
        decButton=QPushButton('-',self)
        saveButton=QPushButton('저장',self)
        quitButton=QPushButton('나가기',self)
        
        fileButton.setGeometry(10,10,100,30)
        paintButton.setGeometry(110,10,100,30)
        cutButton.setGeometry(210,10,100,30)
        incButton.setGeometry(310,10,50,30)
        decButton.setGeometry(360,10,50,30)
        saveButton.setGeometry(410,10,100,30)
        quitButton.setGeometry(510,10,100,30)
        
        fileButton.clicked.connect(self.fileOpenFunction)
        paintButton.clicked.connect(self.paintFunction) 
        cutButton.clicked.connect(self.cutFunction)    
        incButton.clicked.connect(self.incFunction)              
        decButton.clicked.connect(self.decFunction) 
        saveButton.clicked.connect(self.saveFunction)                         
        quitButton.clicked.connect(self.quitFunction)

        self.BrushSiz=5
        self.LColor,self.RColor=(255,0,0),(0,0,255) 
        
    def fileOpenFunction(self):
        fname=QFileDialog.getOpenFileName(self,'Open file','./')
        self.img=cv.imread(fname[0])
        if self.img is None: sys.exit('파일을 찾을 수 없습니다.')  
        
        self.img_show=np.copy(self.img)	 
        cv.imshow('Painting',self.img_show)
        
        self.mask=np.zeros((self.img.shape[0],self.img.shape[1]),np.uint8)
        self.mask[:,:]=cv.GC_PR_BGD
            
    def paintFunction(self):
        cv.setMouseCallback('Painting',self.painting) 
        
    def painting(self,event,x,y,flags,param):
        if event==cv.EVENT_LBUTTONDOWN:   
            cv.circle(self.img_show,(x,y),self.BrushSiz,self.LColor,-1) ## 왼쪽버튼 클릭 파란
            cv.circle(self.mask,(x,y),self.BrushSiz,cv.GC_FGD,-1)
        elif event==cv.EVENT_RBUTTONDOWN: 
            cv.circle(self.img_show,(x,y),self.BrushSiz,self.RColor,-1) ## 오른쪽버튼 클릭 빨간
            cv.circle(self.mask,(x,y),self.BrushSiz,cv.GC_BGD,-1)
        elif event==cv.EVENT_MOUSEMOVE and flags==cv.EVENT_FLAG_LBUTTON:
            cv.circle(self.img_show,(x,y),self.BrushSiz,self.LColor,-1) ## 왼쪽버튼을 클릭이동 파란
            cv.circle(self.mask,(x,y),self.BrushSiz,cv.GC_FGD,-1)
        elif event==cv.EVENT_MOUSEMOVE and flags==cv.EVENT_FLAG_RBUTTON:
            cv.circle(self.img_show,(x,y),self.BrushSiz,self.RColor,-1) ## 오른쪽버튼 클릭이동 빨간 
            cv.circle(self.mask,(x,y),self.BrushSiz,cv.GC_BGD,-1)
    
        cv.imshow('Painting',self.img_show)        
        
    def cutFunction(self):
        background=np.zeros((1,65),np.float64) 
        foreground=np.zeros((1,65),np.float64) 
        cv.grabCut(self.img,self.mask,None,background,foreground,5,cv.GC_INIT_WITH_MASK)
        mask2=np.where((self.mask==2)|(self.mask==0),0,1).astype('uint8')
        self.grabImg=self.img*mask2[:,:,np.newaxis]
        cv.imshow('Scissoring',self.grabImg) 
        
    def incFunction(self):
        self.BrushSiz=min(20,self.BrushSiz+1) 
        
    def decFunction(self):
        self.BrushSiz=max(1,self.BrushSiz-1) 
        
    def saveFunction(self):
        fname=QFileDialog.getSaveFileName(self,'파일 저장','./')
        cv.imwrite(fname[0],self.grabImg)
                
    def quitFunction(self):
        cv.destroyAllWindows()        
        self.close()
                
app=QApplication(sys.argv) 
win=Grabcut() 
win.show()
app.exec_()
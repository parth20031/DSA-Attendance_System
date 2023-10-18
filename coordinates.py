import cv2
import os

cap=cv2.VideoCapture(0)
interface=cv2.imread("sample.jpg")
# interface=cv2.resize(interface,(1240,680))

def draw(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        text=str(x)+","+ str(y)
        cv2.putText(interface,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),5,cv2.LINE_AA) 

cv2.namedWindow(winname="interface")
cv2.setMouseCallback('interface',draw)
while True:
    cv2.imshow("interface",interface)
    if cv2.waitKey(1)==27:
        break
'''
while True:
    ret,frame=cap.read()
    cv2.imshow("frame",frame)
    if cv2.waitKey(1)==27:
        break
'''
cap.release()
cv2.destroyAllWindows()
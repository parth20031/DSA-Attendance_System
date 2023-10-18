import cv2
import os
import pickle
import face_recognition
import cvzone
import numpy as np

cap=cv2.VideoCapture(0)


file=open("encodings.p","rb")
encodes=pickle.load(file)
encodings,RollList=encodes
file.close()
bl=cv2.imread("interface/blackpage.png")
bl=cv2.resize(bl,(1230-800,670-10))
cv2.namedWindow(winname="interface")
while True:
    interface=cv2.imread("interface/as2c.jpeg")
    interface=cv2.resize(interface,(1240,680))

    interface[10:670,800:1230]=bl
    ret,frame=cap.read()
    # frame=cv2.imread("sample.jpg")
    frame=cv2.resize(frame,(640,486))
    interface[116:116+486,127:640+127]=frame
    w=640
    h=486
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame2=frame[0:w/4,0:h/2]
    frame3=frame[w/4:w/2,0:h/2]
    frame4=frame[w/2:3*w/4,0:h/2]
    frame5=frame[3*w/4:w,0:h/2]
    frame6=frame[0:w/4,h/2:h]
    frame7=frame[w/4:w/2,h/2:h]
    frame8=frame[w/2:3*w/4,h/2:h]
    frame9=frame[3*w/4:w,h/2:h]
    frame2=cv2.resize(frame2,(640,486))
    frame3=cv2.resize(frame3,(640,486))
    frame4=cv2.resize(frame4,(640,486))
    frame5=cv2.resize(frame5,(640,486))
    frame6=cv2.resize(frame6,(640,486))
    frame7=cv2.resize(frame7,(640,486))
    frame8=cv2.resize(frame8,(640,486))
    frame9=cv2.resize(frame9,(640,486))

    # ret2,frame2=cap.read()
    # frame2=cv2.resize(frame2,(640,486))
    # interface[116:116+486,127:640+127]=frame2
    # ret3,frame3=cap.read()
    # frame3=cv2.resize(frame3,(640,486))
    # interface[116:116+486,127:640+127]=frame3
 
    
    
    
    interface=cv2.rectangle(interface,(127,116),(640+127,116+486),(0,255,50),3)
    # cv2.imshow("interface",interface)
    
    # frame2=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    # frame3=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    allFacesInFrame=face_recognition.face_locations(frame)
    encodeAllFaces=face_recognition.face_encodings(frame,allFacesInFrame)

    allFacesInFrame2=face_recognition.face_locations(frame2)
    encodeAllFaces2=face_recognition.face_encodings(frame2,allFacesInFrame2)

    allFacesInFrame3=face_recognition.face_locations(frame3)
    encodeAllFaces3=face_recognition.face_encodings(frame3,allFacesInFrame3)

    allFacesInFrame4=face_recognition.face_locations(frame4)
    encodeAllFaces4=face_recognition.face_encodings(frame4,allFacesInFrame4)

    allFacesInFrame5=face_recognition.face_locations(frame5)
    encodeAllFaces5=face_recognition.face_encodings(frame5,allFacesInFrame5)

    allFacesInFrame6=face_recognition.face_locations(frame6)
    encodeAllFaces6=face_recognition.face_encodings(frame6,allFacesInFrame6)

    allFacesInFrame7=face_recognition.face_locations(frame7)
    encodeAllFaces7=face_recognition.face_encodings(frame7,allFacesInFrame7)

    allFacesInFrame8=face_recognition.face_locations(frame8)
    encodeAllFaces8=face_recognition.face_encodings(frame8,allFacesInFrame8)

    allFacesInFrame9=face_recognition.face_locations(frame9)
    encodeAllFaces9=face_recognition.face_encodings(frame9,allFacesInFrame9)
    
    
    
    # allFacesInFrame2=face_recognition.face_locations(frame2)
    # encodeAllFaces2=face_recognition.face_encodings(frame2,allFacesInFrame)
    
    # allFacesInFrame3=face_recognition.face_locations(frame3)
    # encodeAllFaces3=face_recognition.face_encodings(frame3,allFacesInFrame)
    
    matches=[]
    
    for encodes,faceloc in zip(encodeAllFaces,allFacesInFrame):
    # for encodes,encodes2,encodes3,faceloc,faceloc2,faceloc3 in zip(encodeAllFaces,encodeAllFaces2,encodeAllFaces3,allFacesInFrame,allFacesInFrame2,allFacesInFrame3):
        matches=face_recognition.compare_faces(encodings,encodes)
        # matches2=face_recognition.compare_faces(encodings,encodes2)
        # matches3=face_recognition.compare_faces(encodings,encodes3)
        # for match,match2 in zip(matches,matches2):
            # match=match&match2
        # for match,match3 in zip(matches,matches3):
            # match=match&match3
        facedis=face_recognition.face_distance(encodings,encodes)
        # facedis2=face_recognition.face_distance(encodings,encodes2)
        # facedis3=face_recognition.face_distance(encodings,encodes3)

        interface =cv2.rectangle(interface,(127+faceloc[3],116+faceloc[0]),(127+faceloc[1],116+faceloc[2]),(255,255,0),1)
        min=np.argmin(facedis)
        # min2=np.argmin(facedis2)
        # min3=np.argmin(facedis3)
        # min=min(min,min2)
        # min=min(min,min3)
        
        # if min==min2 and min2==min3 and matches[min]:
        if  facedis[min]<0.415:

            font=cv2.FONT_HERSHEY_SIMPLEX
    # index=0
    # print(matches)
    # for comp in matches:
        # if comp==True:
           
            cv2.putText(interface,RollList[min],(835 + ((min)%3)*120 ,38+(min//3)*22),font,0.6,[255,255,255],1,cv2.LINE_AA)
        # index+=1
        
    
    
    cv2.imshow("interface",interface)
    

    
    if cv2.waitKey(1)==27 or cv2.getWindowProperty("interface",0)<0:
        break
cap.release()
cv2.destroyAllWindows()
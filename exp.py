import cv2
import face_recognition
frame=cv2.imread("sample.jpg")
h, w, channels = frame.shape
ph=h//2
pw=w//2
parts = []
allFacesInParts=[]
encodeAllFacesInParts=[]
for i in range(2):
            for j in range(2):
                x_start = j * pw
                x_end = (j + 1) * pw
                y_start = i * ph
                y_end = (i + 1) * ph
                part = frame[y_start:y_end, x_start:x_end]
                height,width,_=part.shape
                
                part=cv2.cvtColor(part,cv2.COLOR_BGR2RGB)
                allFacesInParts.append(face_recognition.face_locations(part))
                encodeAllFacesInParts.append(face_recognition.face_encodings(part,face_recognition.face_locations(part)))
                parts.append(part)
cv2.imshow("i",parts[1])
cv2.waitKey(0)
    

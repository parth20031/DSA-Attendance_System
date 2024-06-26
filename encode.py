import os
import cv2
import pickle
import face_recognition

studentImg=os.listdir("student_details1")

imgList=[]
RollList=[]
for image in studentImg:
    imgList.append(cv2.imread(os.path.join("student_details1",image)))
    RollList.append(image[:-4])
# print(RollList)




def encodes(imgList):
    encodings=[]
    index=0
    for img in imgList:
        print(studentImg[index])
        index+=1
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodings.append(encode)
    return encodings

encodings=encodes(imgList)

encodes=[encodings,RollList]
file=open("encodings2.p","wb")
pickle.dump(encodes,file)
file.close()
print("done")
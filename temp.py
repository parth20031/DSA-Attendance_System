frame=cv2.imread("sample.jpg")
height, width, channels = frame.shape
kernel = np.array([[-1, -1, -1],
                  [-1,  9, -1],
                  [-1, -1, -1]])
sharp_image = cv2.filter2D(frame, -1, kernel)
frame = cv2.resize(frame, (640, 486))
interface[116:116 + 486, 127:640 + 127] = frame
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

allFacesInFrame = face_recognition.face_locations(frame)
encodeAllFaces = face_recognition.face_encodings(frame, allFacesInFrame)

matches = []
for encodes, faceloc in zip(encodeAllFaces, allFacesInFrame):
            # matches = face_recognition.compare_faces(encodings, encodes)
            facedis = face_recognition.face_distance(encodings, encodes)
            interface = cv2.rectangle(interface, (127 + faceloc[3], 116 + faceloc[0]), (127 + faceloc[1], 116 + faceloc[2]), (255, 255, 0), 1)
            
            min_index = np.argmin(facedis)
            if facedis[min_index]<0.415:
            # if matches[min_index]:
                print(facedis[min_index])
                roll_value = RollList[min_index]
                roll_list_values.append(roll_value)
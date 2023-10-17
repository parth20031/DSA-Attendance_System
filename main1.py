# mid working poject 

import cv2
import os
import pickle
import face_recognition
import cvzone
import numpy as np
from pymongo import MongoClient
from datetime import datetime

# Define the starting and ending roll numbers
start_roll_number = 220001001
end_roll_number = 220001082

# List of roll numbers in the specified range
roll_number_list = [str(roll) for roll in range(start_roll_number, end_roll_number + 1)]

# Add the extra roll numbers
extra_roll_numbers = ["220002018", "220002028","220002063","220002081"]
roll_number_list.extend(extra_roll_numbers)

# Connect to the MongoDB database
client = MongoClient('mongodb://localhost:27017/')  # Replace with your MongoDB connection string
db = client['attendance']  # Set the database name to "attendance"
today_date = datetime.now().strftime("%Y-%m-%d")  # Get the current date in YYYY-MM-DD format
collection = db[today_date]  # Use the current date as the collection name

# Create a new collection name for today_data_attendance
today_data_attendance_collection_name = today_date + "_attendance"
today_data_attendance_collection = db[today_data_attendance_collection_name]

# Add a "remark" column and mark all entries as "Present"
today_data_attendance_collection.update_many({}, {"$set": {"remark": "Present"}})

# Create the "student_info" collection and insert the roll numbers
student_info_collection = db['student_info']
for roll_number in roll_number_list:
    document = {"roll_number": roll_number}
    student_info_collection.insert_one(document)

print("Inserted student info into the 'student_info' collection.")

# Create a new collection name for attendance_all
attendance_all_collection_name = today_date + "_attendance_all"
attendance_all_collection = db[attendance_all_collection_name]

# Iterate through the roll numbers and mark them as absent with a timestamp
for roll_number in roll_number_list:
    document = {
        "roll_number": roll_number,
        "timestamp": datetime.now(),
        "remark": "Absent"
    }
    attendance_all_collection.insert_one(document)

print(f"Inserted roll numbers into the '{attendance_all_collection_name}' collection with 'Absent' status.")

cap = cv2.VideoCapture(0)

file = open("encodings.p", "rb")
encodes = pickle.load(file)
encodings, RollList = encodes
file.close()

# Create an empty list to store the RollList[min] values
roll_list_values = []
interface = cv2.imread("interface/as2c.jpeg")
interface = cv2.resize(interface, (1240, 680))
bl = cv2.imread("interface/blackpage.png")
bl = cv2.resize(bl, (1230 - 800, 670 - 10))
interface[10:670, 800:1230] = bl


cv2.namedWindow(winname="interface")


interface = cv2.rectangle(interface, (127, 116), (640 + 127, 116 + 486), (0, 255, 50), 3)

while True:
    ret, frame = cap.read()
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

            # Check if the roll_value is not already in today_data_attendance
            existing_doc = today_data_attendance_collection.find_one({"roll_value": roll_value})
            if existing_doc is None:
                # If not found, insert it into the new collection with timestamp and mark as "Present"
                document = {"roll_value": roll_value, "timestamp": datetime.now(), "remark": "Present"}
                today_data_attendance_collection.insert_one(document)
                
                # Check if the roll_value is also present in attendance_all
                existing_attendance_doc = attendance_all_collection.find_one({"roll_number": roll_value})
                if existing_attendance_doc is None:
                    # If not found, insert it into attendance_all with the same timestamp and mark as "Present"
                    attendance_all_collection.insert_one(document)
                else:
                    # If found, update the timestamp and mark as "Present"
                    attendance_all_collection.update_one({"roll_number": roll_value}, {"$set": {"timestamp": datetime.now(), "remark": "Present"}})

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(interface, roll_value, (835 + ((min_index) % 3) * 120, 38 + (min_index // 3) * 22), font, 0.6, [255, 255, 255], 1, cv2.LINE_AA)
            print(roll_value)
    cv2.imshow("interface", interface)

    if cv2.waitKey(1) == 27 :
        break
print(roll_list_values)

cap.release()
cv2.destroyAllWindows()

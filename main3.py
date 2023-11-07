from flask import Flask, render_template, request, jsonify,send_file,send_from_directory
import cv2
import os
import pickle
import face_recognition
import cvzone
import numpy as np
from pymongo import MongoClient
from datetime import datetime
import pymongo
from bson import json_util
import pandas as pd
from io import BytesIO


app = Flask(__name__)

# Define the folder where you'll store uploaded images.
UPLOAD_FOLDER = 'student_details'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)



# MongoDB connection configuration
mongo_uri = 'mongodb://localhost:27017'  # Replace with your MongoDB connection string
client = pymongo.MongoClient(mongo_uri)
db = client['attendance']

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
    # Check if a document with the same roll number exists
    existing_student = student_info_collection.find_one({"roll_number": roll_number})

    if existing_student:
        # Document with the same roll number already exists
        print(f"Student with roll number {roll_number} already exists in the collection.")
    else:
        # Document with the same roll number doesn't exist, so insert it
        document = {"roll_number": roll_number}
        student_info_collection.insert_one(document)
        print(f"Inserted student with roll number {roll_number} into the 'student_info' collection.")


# Create a new collection name for attendance_all
attendance_all_collection_name = today_date + "_attendance_all"
attendance_all_collection = db[attendance_all_collection_name]

# Iterate through the roll numbers and mark them as absent with a timestamp
for roll_number in roll_number_list:
    # Check if a document with the same roll number exists
    existing_attendance = attendance_all_collection.find_one({"roll_number": roll_number})
    if existing_attendance:
        print(f"Updated attendance for roll number {roll_number} in the '{attendance_all_collection_name}' collection.")
    else:
        # Document with the same roll number doesn't exist, so insert it
        document = {
            "roll_number": roll_number,
            "timestamp": datetime.now(),
            "remark": "Absent"
        }
        attendance_all_collection.insert_one(document)
        print(f"Inserted attendance for roll number {roll_number} into the '{attendance_all_collection_name}' collection with 'Absent' status.")

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'image' in request.files:
            image = request.files['image']
            if image.filename != '':
                image.save(os.path.join(app.config['UPLOAD_FOLDER'], image.filename))
                return "Image uploaded successfully!"

    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/interface')
def interface():
    # cap = cv2.VideoCapture(0)

    file = open("encodings1.p", "rb")
    encodes = pickle.load(file)
    encodings, RollList = encodes
    file.close()

    # Create an empty list to store the RollList[min] values
    roll_list_values = []
    interface = cv2.imread("newinterface.jpg")
    # interface = cv2.resize(interface, (1240, 680))
    # bl = cv2.imread("interface/blackpage.png")
    # bl = cv2.resize(bl, (1230 - 800, 670 - 10))
    # interface[10:670, 800:1230] = bl


    cv2.namedWindow(winname="interface")


    # interface = cv2.rectangle(interface, (127, 116), (640 + 127, 116 + 486), (0, 255, 50), 3)

    while True:
        # ret, frame = cap.read()
        frame=cv2.imread("sample7.png")
        h, w, channels = frame.shape
        ph=h
        pw=w//2
        parts = []
        allFacesInParts=[]
        encodeAllFacesInParts=[]
        for i in range(2):
            
                x_start = i * pw
                x_end = (i + 1) * pw
                y_start = 0
                y_end =ph
                part = frame[y_start:y_end, x_start:x_end]
                he,we,_=part.shape
                part=cv2.resize(part,(2*we,2*he))

                # part=cv2.resize(part,(640,486))
                part=cv2.cvtColor(part,cv2.COLOR_BGR2RGB)
                allFacesInParts.append(face_recognition.face_locations(part))
                encodeAllFacesInParts.append(face_recognition.face_encodings(part,face_recognition.face_locations(part)))
                parts.append(part)
        


        # allFacesInFrame = face_recognition.face_locations(frame)
        # encodeAllFaces = face_recognition.face_encodings(frame, allFacesInFrame)


        for i in range(len(allFacesInParts)):
            for encodes,faceloc in zip(encodeAllFacesInParts[i],allFacesInParts[i]):
                facedis = face_recognition.face_distance(encodings, encodes)

                min_index=np.argmin(facedis)
                if facedis[min_index]<0.6:
                    parts[i]=cv2.rectangle(parts[i],(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,255,0),4)
                    print(facedis[min_index])
                    roll_value=RollList[min_index][:-2]
                    roll_list_values.append(roll_value)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(interface, roll_value, (835 + (((int(RollList[min_index][7:-2]))-1) % 3) * 120, 38 + ((int(RollList[min_index][7:-2])-1) // 3) * 22), font, 0.6, [255, 255, 255], 1, cv2.LINE_AA)
                existing_doc = today_data_attendance_collection.find_one({"roll_number": roll_value})
                if existing_doc is None:
                    # If not found, insert it into the new collection with timestamp and mark as "Present"
                    document = {"roll_number": roll_value, "timestamp": datetime.now(), "remark": "Present"}
                    today_data_attendance_collection.insert_one(document)
                    
                    # Check if the roll_value is also present in attendance_all
                    existing_attendance_doc = attendance_all_collection.find_one({"roll_number": roll_value})
                    if existing_attendance_doc is None:
                        # If not found, insert it into attendance_all with the same timestamp and mark as "Present"
                        attendance_all_collection.insert_one(document)
                    else:
                        # If found, update the timestamp and mark as "Present"
                        attendance_all_collection.update_one({"roll_number": roll_value}, {"$set": {"timestamp": datetime.now(), "remark": "Present"}})  
                    

                print(roll_value)
        
        for i in range(2):

            parts[i]=cv2.resize(parts[i],(pw,ph))

        frame = np.zeros((h, w, 3), dtype=np.uint8)
        part_index = 0
        for i in range(2):
            #sample
                x_start = i * pw
                x_end = (i + 1) * pw
                y_start = 0
                y_end =  ph

              
                frame[y_start:y_end, x_start:x_end] = parts[part_index]
                part_index += 1
        frame = cv2.resize(frame, (640, 486))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        interface[116:116 + 486, 127:640 + 127] = frame
        # for encodes, faceloc in zip(encodeAllFaces, allFacesInFrame):
        #     # matches = face_recognition.compare_faces(encodings, encodes)
        #     facedis = face_recognition.face_distance(encodings, encodes)
        #     interface = cv2.rectangle(interface, (127 + faceloc[3], 116 + faceloc[0]), (127 + faceloc[1], 116 + faceloc[2]), (255, 255, 0), 1)
            
        #     min_index = np.argmin(facedis)
        #     if facedis[min_index]<0.415:
        #     # if matches[min_index]:
        #         print(facedis[min_index])
        #         roll_value = RollList[min_index]
        #         roll_list_values.append(roll_value)

        #         # Check if the roll_value is not already in today_data_attendance
        #         existing_doc = today_data_attendance_collection.find_one({"roll_number": roll_value})
        #         if existing_doc is None:
        #             # If not found, insert it into the new collection with timestamp and mark as "Present"
        #             document = {"roll_number": roll_value, "timestamp": datetime.now(), "remark": "Present"}
        #             today_data_attendance_collection.insert_one(document)
                    
        #             # Check if the roll_value is also present in attendance_all
        #             existing_attendance_doc = attendance_all_collection.find_one({"roll_number": roll_value})
        #             if existing_attendance_doc is None:
        #                 # If not found, insert it into attendance_all with the same timestamp and mark as "Present"
        #                 attendance_all_collection.insert_one(document)
        #             else:
        #                 # If found, update the timestamp and mark as "Present"
        #                 attendance_all_collection.update_one({"roll_number": roll_value}, {"$set": {"timestamp": datetime.now(), "remark": "Present"}})

        #         font = cv2.FONT_HERSHEY_SIMPLEX
        #         cv2.putText(interface, roll_value, (835 + ((min_index) % 3) * 120, 38 + (min_index // 3) * 22), font, 0.6, [255, 255, 255], 1, cv2.LINE_AA)
        #         print(roll_value)
        cv2.imshow("interface", interface)

        if cv2.waitKey(1) == 27 :
            break
    print(roll_list_values)

    # cap.release()
    cv2.destroyAllWindows()



    return render_template('index.html')

@app.route('/', methods=['POST', 'GET'])
def home():

#--------------------------------- info1--------------------------------------------------------------------------------
  if 'input_date' in request.form:
    if request.method == 'POST':
        input_date = request.form['input_date']

      
       # Assuming your collection is named "{input_date}_attendance_all"
        collection_name = f"{input_date}_attendance_all"

        # Get the specified collection
        collection = db[collection_name]

        # Find all documents in the collection
        result = list(collection.find())

        if result:
            # Convert the result to a DataFrame
            df = pd.DataFrame(result)

            # Drop the first (index 0) and third (index 2) columns by integer position
            df = df.drop(df.columns[[0, 2]], axis=1)

            # Create a BytesIO object to save the Excel file
            excel_output = BytesIO()
            writer = pd.ExcelWriter(excel_output, engine='xlsxwriter')
            df.to_excel(writer, sheet_name='Sheet1', index=False)
            writer.close()  # Close the ExcelWriter

            excel_output.seek(0)

            # Send the Excel file as a downloadable attachment with the input_date variable as the filename
            return send_file(excel_output, as_attachment=True, download_name=f'{input_date}_attendance_all.xlsx')
            
        else :
            print("The attendance is not prsent of that day")
            return render_template('noattendance.html')
    
    # ----------------------------------info2----------------------------------------------------
  elif 'input_roll' in request.form:
    if request.method == 'POST':
        input_roll = request.form['input_roll']

        # Find collections with the specified roll number marked as "Present"
        matching_collections = []
        all_collections = db.list_collection_names()

        for collection_name in all_collections:
            # Skip collections that don't end with "_attendance_all"
            if not collection_name.endswith("_attendance_all"):
                continue

            # Split the collection name and take the part before "_attendance_all"
            collection_prefix = collection_name.split("_attendance_all")[0]

            collection = db[collection_name]
            # Check if the roll number is present and marked as "Present" in the collection
            if collection.find_one({"roll_number": input_roll, "remark": "Present"}):
                matching_collections.append(collection_prefix)

        # Return the matching collection names
        return render_template('matching_collections.html', collections=matching_collections)
  return render_template('index.html')





if __name__ == "__main__":
    app.run(debug=True,port=3000)


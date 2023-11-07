from flask import Flask, render_template, request, jsonify,send_file,send_from_directory,url_for
from flask import redirect
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
import xlsxwriter
from io import BytesIO
from pandas import DataFrame
from collections import defaultdict
from flask_mail import Mail
from flask_mail import Message


app = Flask(__name__)

mail = Mail(app)


# Define the folder where you'll store uploaded images.
UPLOAD_FOLDER = 'student_details'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define a global variable to store eligible roll numbers
eligible_roll_numbers = []


def access_database(subject_name):
    client = MongoClient('mongodb://localhost:27017/') 
    db = client[subject_name]
    return db

def generate_roll_numberlist():
    # Define the starting and ending roll numbers
    start_roll_number = 220001001
    end_roll_number = 220001082

        # List of roll numbers in the specified range
    roll_number_list = [str(roll) for roll in range(start_roll_number, end_roll_number + 1)]

        # Add the extra roll numbers
    extra_roll_numbers = ["220002018", "220002028","220002063","220002081"]
    roll_number_list.extend(extra_roll_numbers)
    return roll_number_list

    # Connect to the MongoDB database
client = MongoClient('mongodb://localhost:27017/')  # Replace with your MongoDB connection string
# db = client['attendance']  # Set the database name to "attendance"


dbs= client['subjects']
subjects_collection = dbs['subject_collection']

def load_subjects_from_database():
    return [subject['name'] for subject in subjects_collection.find()]

# Load subjects from the database into the subjects list when the server starts
subjects = load_subjects_from_database()


def create_or_get_database(db_name):
    # Connect to your MongoDB server
    client = MongoClient('mongodb://localhost:27017')  # Replace with your MongoDB connection string
    
    # Check if the database already exists
    if db_name in client.list_database_names():
        return client[db_name]  # Return the existing database
    else:
        return client[db_name]  # Create a new database and return it
    
def create_or_get_database_with_collection(db_name, collection_name, document):
    # Connect to your MongoDB server
    client = MongoClient('mongodb://localhost:27017')  # Replace with your MongoDB connection string
    
    # Check if the database already exists
    if db_name in client.list_database_names():
        db = client[db_name]  # Return the existing database
    else:
        db = client[db_name]  # Create a new database
        # Add a collection to the new database
        db.create_collection(collection_name)
    
    # Insert the document into the collection
    db[collection_name].insert_one(document)
    
#     return db

# Define a function to create or get a collection with the specified name
def create_or_get_collection(db, collection_name):
    if collection_name not in db.list_collection_names():
        return db[collection_name]
    else:
        return db[collection_name]
    
def today_date():
    today_date = datetime.now().strftime("%Y-%m-%d")  # Get the current date in YYYY-MM-DD format
    return today_date

# Define a function to create and initialize the "student_info" collection
def create_student_info_collection(db):
    student_info_collection = db['student_info']
    roll_number_list=generate_roll_numberlist()
    for roll_number in roll_number_list:
        # Check if a document with the same roll number exists
        existing_student = student_info_collection.find_one({"roll_number": roll_number})

        if existing_student is None:
        
            # Document with the same roll number doesn't exist, so insert it
            document = {"roll_number": roll_number}
            student_info_collection.insert_one(document)
            print(f"Inserted student with roll number {roll_number} into the 'student_info' collection.")



# collection = db[today_date]  # Use the current date as the collection name
# Create a new collection name for today_data_attendance

def create_attendance_collection(db):
    # today_date=today_date()
    today_date = datetime.now().strftime("%Y-%m-%d")
    today_data_attendance_collection_name = today_date + "_attendance"
    today_data_attendance_collection = db[today_data_attendance_collection_name]

    data_to_insert = {
        "random": "-2",
        "timestamp": datetime.now(),
        # "remark": "Present"
    }
    
    # Insert the data into the collection
    today_data_attendance_collection.insert_one(data_to_insert)

    # Add a "remark" column and mark all entries as "Present"
    today_data_attendance_collection.update_many({}, {"$set": {"remark": "Present"}})
    print(today_data_attendance_collection)  
    return today_data_attendance_collection


def create_attendance_all_collection(db):
    # Create a new collection name for attendance_all
    roll_number_list = generate_roll_numberlist()
    # today_date = today_date()
    today_date = datetime.now().strftime("%Y-%m-%d")
    attendance_all_collection_name = today_date + "_attendance_all"
    attendance_all_collection = db[attendance_all_collection_name]
    print(attendance_all_collection)  
# sample
    # Clear the existing data in the collection
    attendance_all_collection.delete_many({})

    # Iterate through the roll numbers and mark them as absent with a timestamp
    for roll_number in roll_number_list:
        # Insert each roll number as "Absent" with a timestamp
        document = {
            "roll_number": roll_number,
            "timestamp": datetime.now(),
            "remark": "Absent"
        }
        attendance_all_collection.insert_one(document)
    return attendance_all_collection


@app.route('/')
def home():    
    return render_template('index2.html', subjects=subjects)


@app.route('/add_subject', methods=['POST'])
def add_subject():
    # subject_name = request.form.get('subject_name')
    # if subject_name:
    #     subjects.append(subject_name)
    # return redirect(url_for('home'))
    if request.method == 'POST':
        subject_name = request.form['subject_name']
        if subject_name not in subjects:
            subjects_collection = dbs['subject_collection']  # Use a specific collection within the 'subjects' database
            existing_subject = subjects_collection.find_one({"name": subject_name})
            if existing_subject is None:
                subjects_collection.insert_one({'name': subject_name})
            subjects.append(subject_name)
            db_name = subject_name  # Replace with your desired database name
            collection_name = "my_collection"  # Replace with your desired collection name
            document = {"key": "value"}  # Replace with the document you want to insert
            create_or_get_database_with_collection(db_name, collection_name, document)

    return redirect(url_for('home'))


@app.route('/<subjects>/start', methods=['POST', 'GET'])
def start(subjects):   
    return render_template('index.html',subjects=subjects)


def extract_date_from_collection(collection_name):
    return collection_name.split('_attendance_all')[0]


@app.route('/<subjects>/docs', methods=['POST', 'GET'])
def docs(subjects):
    db=access_database(subjects)
    # List all collection names in the database
    all_collections = db.list_collection_names()

    # Filter and count collections ending with "attendance_all"
    attendance_collections = [collection for collection in all_collections if collection.endswith("_attendance_all")]
    number_of_collections = len(attendance_collections)

    # Print the number of "attendance_all" collections
    print(f'Number of "attendance_all" collections: {number_of_collections}')

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
                return send_file(excel_output, as_attachment=True, download_name=f'{input_date}_{subjects}_attendance_all.xlsx')
                
            else :
                print("The attendance is not present of that day")
                return render_template('noattendance.html')
    
    # ----------------------------------info2----------------------------------------------------
#   elif 'input_roll' in request.form:
    elif 'input_roll' in request.form:
            # Handle input_roll case
            if request.method == 'POST':
                input_roll = request.form['input_roll']

                # Get all collection names that end with "_attendance_all"
                all_collections = db.list_collection_names()
                matching_dates = []  # To store dates where the student was present

                for collection_name in all_collections:
                    if collection_name.endswith('_attendance_all'):
                        collection = db[collection_name]
                        if collection.find_one({"roll_number": input_roll, "remark": "Present"}):
                            date = extract_date_from_collection(collection_name)
                            matching_dates.append(date)

                if matching_dates:
                    # Create a DataFrame to store the matching dates
                    data = {'Dates': matching_dates}
                    df = DataFrame(data)

                    # Create an Excel file from the DataFrame
                    excel_output = BytesIO()
                    writer = pd.ExcelWriter(excel_output, engine='xlsxwriter')
                    df.to_excel(writer, sheet_name='Sheet1', index=False)
                    writer.close()
                    excel_output.seek(0)

                    # Return the Excel file as a downloadable attachment
                    return send_file(excel_output, as_attachment=True, download_name=f'{input_roll}_{subjects}_attendance_dates.xlsx')

                else:
                    return render_template('noattendance.html')
    elif 'input_percent' in request.form:
            # Handle input_percent case
            if request.method == 'POST':
                input_percent = request.form['input_percent']
                print(input_percent) 
                print(f'Number of "attendance_all" collections: {number_of_collections}')
                number_of_classes = int((int(input_percent) / 100) * number_of_collections)
                print(f'Number of classes based on {input_percent}%: {number_of_classes}')
                # Create a dictionary to count the appearances of roll_numbers
                roll_number_counts = defaultdict(int)

                # Ensure 'all_collections' is defined here
                all_collections = db.list_collection_names()

                # Create a dictionary to count the appearances of roll_numbers
                roll_number_counts = defaultdict(int)

                # Iterate through all "attendance_all" collections
                for collection_name in all_collections:
                    if collection_name.endswith("_attendance_all"):
                        collection = db[collection_name]

                        # Find all documents in the collection
                        all_students = collection.find({})

                        # Increment the count of each roll_number
                        for student in all_students:
                            roll_number = student["roll_number"]
                            roll_number_remark = student.get("remark", "")  # Use get to handle cases where "remark" is missing

                            if roll_number_remark == "Present":
                                roll_number_counts[roll_number] += 1

                            elif roll_number_remark == "Absent":
                                roll_number_counts[roll_number] += 0

                print("Roll Number Counts:")
                for roll_number, count in roll_number_counts.items():
                    print(f"Roll Number: {roll_number}, Count: {count}")

                # Print the counts for roll numbers that have counts less than number_of_classes
                print("Roll Numbers with Counts Less than Number of Classes:")
                for roll_number, count in roll_number_counts.items():
                    if count < number_of_classes:
                        print(f"Roll Number: {roll_number}, Count: {count}")

                # Filter roll_numbers with counts less than number_of_classes
                # Create a list to store eligible roll numbers
                global eligible_roll_numbers
                eligible_roll_numbers = [roll_number for roll_number, count in roll_number_counts.items() if count < number_of_classes]

                # Print the eligible_roll_numbers in the console
                print("Eligible Roll Numbers:")
                print(eligible_roll_numbers)
            return redirect('new_route')

    return render_template('docs.html',subjects=subjects)


@app.route('/<subjects>/new_route', methods=['GET'])
def new_route(subjects):
    global eligible_roll_numbers  # Access the global variable
    return render_template('new_route.html', eligible_roll_numbers=eligible_roll_numbers,subjects=subjects)


# Configure email settings for Gmail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465  # Port for secure SSL/TLS
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USERNAME'] = 'cse220001057@iiti.ac.in'  # Your Gmail email address
# app.config['MAIL_PASSWORD'] =   # Your Gmail email password
app.config['MAIL_DEFAULT_SENDER'] = 'cse220001057@iiti.ac.in'  # Default sender


@app.route('/send_email', methods=['GET', 'POST'])
def send_email():
    if request.method == 'POST':
        recipient = 'deshmukhparth293@gmail.com'  # Replace with the recipient's email address
        subject = 'Hello, User!'
        message = 'This is a test email sent from your Flask app.'

        msg = Message(subject, sender='your_email@gmail.com', recipients=[recipient])
        msg.body = message

        mail.send(msg)
    return 'Email sent successfully!'


@app.route('/<subjects>/interface')
def interface(subjects):
    db=access_database(subjects)
    create_student_info_collection(db)
    today_data_attendance_collection=create_attendance_collection(db)
    attendance_all_collection=create_attendance_all_collection(db)
    # cap = cv2.VideoCapture(0)

    file = open("encodings2.p", "rb")
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
        # ret, frame = cap.read()
        frame=cv2.imread("sample7.jpg")
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
        part=frame[0:ph,w//3:2*w//3]
        he,we,_=part.shape
        part=cv2.resize(part,(2*we,2*he))

        # part=cv2.resize(part,(640,486))
        part=cv2.cvtColor(part,cv2.COLOR_BGR2RGB)
        allFacesInParts.append(face_recognition.face_locations(part))
        encodeAllFacesInParts.append(face_recognition.face_encodings(part,face_recognition.face_locations(part)))
        parts.append(part)

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
            
                x_start = i * pw
                x_end = (i + 1) * pw
                y_start = 0
                y_end =  ph

              
                frame[y_start:y_end, x_start:x_end] = parts[part_index]
                part_index += 1
        frame = cv2.resize(frame, (640, 486))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        interface[116:116 + 486, 127:640 + 127] = frame
        cv2.imshow("interface", interface)

        if cv2.waitKey(1) == 27 :
            break
    print(roll_list_values)

    # cap.release()
    cv2.destroyAllWindows()
    return render_template('index.html',subjects=subjects)


image=[]
@app.route('/<subjects>/upload', methods=['GET', 'POST'])
def upload_file(subjects):
    if request.method == 'POST':
        if 'image' in request.files:
            image = request.files['image']
            if image.filename != '':
                image.save(os.path.join(app.config['UPLOAD_FOLDER'], image.filename))
                return "Image uploaded successfully!"

    return render_template('upload.html')


@app.route('/<subjects>/uploads/<filename>')
def uploaded_file(subjects,filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
    app.run(debug=True,port=3000)


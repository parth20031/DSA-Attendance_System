# pip install Flask opencv-python-headless face-recognition cvzone pymongo pandas xlsxwriter Flask-Mail requests google-auth google-auth-oauthlib
# pip install virtualenv
# virtualenv venv
# pip install -r requirements.txt


from flask import Flask, render_template, request, jsonify,send_file,send_from_directory,url_for,session,abort
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
import os
import pathlib
import requests
from flask import Flask, session, abort, redirect, request
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
from pip._vendor import cachecontrol
import google.auth.transport.requests
#interface
import base64
from functools import wraps
app = Flask(__name__)

app.config['SECRET_KEY'] = "tsfyguaistyatuis589566875623568956"
app.config['MAIL_SERVER'] = "smtp.googlemail.com"
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = "deshmukhparth293@gmail.com"
app.config['MAIL_PASSWORD'] = "toix xman fsba ppqs"

mail = Mail(app)

@app.route('/edit_mail/<eligible_roll_numbers>')
def compose_email_form(eligible_roll_numbers):
    return render_template('email_form.html',eligible_roll_numbers=eligible_roll_numbers)

@app.route('/send_email/<email>', methods=['POST'])
def send_email(email):
    email_title = request.form.get('email_title')
    email_body = request.form.get('email_body')

    msg = Message(email_title, sender="noreply@.com", recipients=[email])
    msg.html = email_body

    try:
        mail.send(msg)
        return "Email sent successfully."
    except Exception as e:
        print(e)
        return "The email was not sent."


allowed_emails = ["cse220001057@iiti.ac.in", "cse220001056@iiti.ac.in","deshmukhparth293@gmail.com","neerupam.26@gmail.com"]

app.secret_key = "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"

os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

GOOGLE_CLIENT_ID = "113932707724-a2g7hh9eos9n6bm1b523741blucaiq5k.apps.googleusercontent.com"
client_secrets_file = os.path.join(pathlib.Path(__file__).parent, "client_secret_113932707724-a2g7hh9eos9n6bm1b523741blucaiq5k.apps.googleusercontent.com.json")

flow = Flow.from_client_secrets_file(
    client_secrets_file=client_secrets_file,
    scopes=["https://www.googleapis.com/auth/userinfo.profile", "https://www.googleapis.com/auth/userinfo.email", "openid"],
    redirect_uri="http://127.0.0.1:3000/callback"
)

# Simulate user authentication
is_authenticated = False

def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not is_authenticated:
            return redirect("/login")  # Redirect to the login page
        return func(*args, **kwargs)
    return wrapper
#delete

@app.route("/login")
def login():
    authorization_url, state = flow.authorization_url()
    session["state"] = state
    return redirect(authorization_url)

@app.route("/callback")
def callback():
    flow.fetch_token(authorization_response=request.url)

    if not session["state"] == request.args["state"]:
        abort(500)  # State does not match!

    credentials = flow.credentials
    request_session = requests.session()
    cached_session = cachecontrol.CacheControl(request_session)
    token_request = google.auth.transport.requests.Request(session=cached_session)

    id_info = id_token.verify_oauth2_token(
        id_token=credentials._id_token,
        request=token_request,
        audience=GOOGLE_CLIENT_ID,
        clock_skew_in_seconds=10  # 5 minutes tolerance
    )
    global is_authenticated 
    is_authenticated = True
    session["google_id"] = id_info.get("sub")
    session["email"] = id_info.get("email")
    # print(session["google_id"])
    session["name"] = id_info.get("name")
    print(session["name"])
    if session["email"] not in allowed_emails:
        return "Access Denied: Your email is not authorized to log in."
    # Fetch the user's profile image from the Google People API
    user_info_url = "https://people.googleapis.com/v1/people/me?personFields=photos"
    headers = {"Authorization": f"Bearer {credentials.token}"}
    user_info_response = requests.get(user_info_url, headers=headers)

    if user_info_response.status_code == 200:
        user_info = user_info_response.json()
        print("try:",user_info)
        if "photos" in user_info:
            profile_image_url = user_info["photos"][0]["url"]
            session["profile_image_url"] = profile_image_url
        else:
            session["profile_image_url"] = None
    else:
        session["profile_image_url"] = None

    return redirect("/protected_area")


@app.route("/logout")
def logout():
    session.clear()
    global is_authenticated 
    is_authenticated = False
    return redirect("/explore")


@app.route("/protected_area")
# @login_is_required
def protected_area():
    return redirect("/explore")    

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


client = MongoClient('mongodb://localhost:27017/') 
dbs= client['subjects']
subjects_collection = dbs['subject_collection']


# # Helper function to load subjects from the 'subjects' collection
# def load_subjects_from_database():
#     client = MongoClient('mongodb://localhost:27017/')  # Replace with your MongoDB connection string
#     db = client['subjects']  # Set the database name to "subjects"
#     subjects_collection = db['subject_collection']
#     return [subject['name'] for subject in subjects_collection.find()]

def load_subjects_from_database():
    client = MongoClient('mongodb://localhost:27017/')  # Replace with your MongoDB connection string
    db = client['subjects']  # Set the database name to "subjects"
    subjects_collection = db['subject_collection']
    return [subject['name'] for subject in subjects_collection.find()]

# Load subjects from the database into the subjects list when the server starts
subjects = load_subjects_from_database()


# Helper function to update the subject name in the 'subjects' collection
def update_subject_name_in_subjects(subject_name, new_name):
    client = MongoClient('mongodb://localhost:27017/')  # Replace with your MongoDB connection string
    db = client['subjects']  # Set the database name to "subjects"
    subjects_collection = db['subject_collection']

    # Rename the subject in the 'subjects' collection
    subjects_collection.update_one({'name': subject_name}, {'$set': {'name': new_name}})

    # Use the existing data transfer function to copy data from the old subject collection to the new one
    # copy_data_to_new_subject(subject_name, new_name)

# Helper function to create or get a database with a collection and insert a document
def create_or_get_database_with_collection(db_name, collection_name, document):
    client = MongoClient('mongodb://localhost:27017')  # Replace with your MongoDB connection string
    
    if db_name in client.list_database_names():
        db = client[db_name]
    else:
        db = client[db_name]
        db.create_collection(collection_name)
    
    db[collection_name].insert_one(document)

def copy_all_collections(old_db_name, new_db_name):
    client = MongoClient('mongodb://localhost:27017')  # Replace with your MongoDB connection string

    # Access the old and new databases
    old_db = client[old_db_name]
    new_db = client[new_db_name]

    # List the collections in the old database
    collections = old_db.list_collection_names()

    for collection_name in collections:
        source_collection = old_db[collection_name]
        target_collection = new_db[collection_name]

        # Copy documents from the source collection to the target collection
        for document in source_collection.find():
            target_collection.insert_one(document)
    
    # Drop the old database after copying
    client.drop_database(old_db_name)
#video
@app.route('/edit_subject', methods=['POST'])
@login_required
def edit_subject():
    if request.method == 'POST':
        # Get the current subject name and the new subject name from the form
        current_name = request.form.get('current_name')
        new_name = request.form.get('new_name')

        # Update the subject name in the 'subjects' collection and copy data
        update_subject_name_in_subjects(current_name, new_name)

        collection_name = "my_collection"  # Replace with your desired collection name
        document = {"key": "value333"} 
        create_or_get_database_with_collection(new_name, collection_name, document)

        copy_all_collections(current_name, new_name)

        # Now update the subjects list in your application
        subjects = load_subjects_from_database()

        return render_template('index2.html', subjects=subjects, is_authenticated=is_authenticated)

    # Handle errors or redirection if the subject doesn't exist
    return redirect('/')

def create_or_get_database(db_name):
    # Connect to your MongoDB server
    client = MongoClient('mongodb://localhost:27017')  # Replace with your MongoDB connection string
    
    # Check if the database already exists
    if db_name in client.list_database_names():
        return client[db_name]  # Return the existing database
    else:
        return client[db_name]  # Create a new database and return it
    #video
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
#video
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


def create_or_get_attendance_all_collection(db):
    today_date = datetime.now().strftime("%Y-%m-%d")
    attendance_all_collection_name = today_date + "_attendance_all"
    
    # Check if the collection already exists
    if attendance_all_collection_name not in db.list_collection_names():
        # Collection doesn't exist, so create and initialize it
        attendance_all_collection = db[attendance_all_collection_name]
        
        # Iterate through the roll numbers and mark them as absent with a timestamp
        roll_number_list = generate_roll_numberlist()
        for roll_number in roll_number_list:
            document = {
                "roll_number": roll_number,
                "timestamp": datetime.now(),
                "remark": "Absent"
            }
            attendance_all_collection.insert_one(document)
        
    else:
        # Collection already exists, so just access it
        attendance_all_collection = db[attendance_all_collection_name]
    
    return attendance_all_collection

@app.route('/')
def welcome():
    return render_template('video.html')

@app.route('/explore')
# @login_is_required
def home():    
    subjects = load_subjects_from_database()
    profile_image_url = session.get("profile_image_url")
    return render_template('index2.html', subjects=subjects, is_authenticated=is_authenticated,profile_image_url=profile_image_url)

@app.route('/delete_subject', methods=['POST'])
@login_required
def delete_subject():
    if request.method == 'POST':
        # Get the subject name from the form
        subject_name = request.form.get('subject_name')

        # Connect to the MongoDB database
        client = MongoClient('mongodb://localhost:27017/')  # Replace with your MongoDB connection string
        db = client['subjects']  # Set the database name to "subjects"
        subjects_collection = db['subject_collection']

        # Check if the subject exists in your database
        subject = subjects_collection.find_one({'name': subject_name})
        if subject:
            # If it exists, remove it from the database
            subjects_collection.delete_one({'name': subject_name})

            client.drop_database(subject_name)
            if subject_name in subjects:
                subjects.remove(subject_name)
            return redirect('/explore')  # Redirect to the index page or another appropriate page

    # Handle errors or redirection if the subject doesn't exist
    return redirect('/explore')


@app.route('/add_subject', methods=['POST'])
@login_required
def add_subject():
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
@login_required
def start(subjects):   
    return render_template('index.html',subjects=subjects)

@app.route('/<subjects>/biinterface')
@login_required
def biinterface(subjects):   
    return render_template('biinterface.html',subjects=subjects)

def extract_date_from_collection(collection_name):
    return collection_name.split('_attendance_all')[0]




@app.route('/<subjects>/new_route', methods=['GET'])
@login_required
def new_route(subjects):
    global eligible_roll_numbers  # Access the global variable
    return render_template('new_route.html', eligible_roll_numbers=eligible_roll_numbers,subjects=subjects)

#biinterface
roll_list_values = []


# # roll_list_values = []

@app.route('/<subjects>/interface')
@login_required

def interface(subjects):
    db=access_database(subjects)
    create_student_info_collection(db)
    today_data_attendance_collection=create_attendance_collection(db)
    attendance_all_collection = create_or_get_attendance_all_collection(db)
    # cap = cv2.VideoCapture(0)

    file = open("encodings2.p", "rb")
    encodes = pickle.load(file)
    encodings, RollList = encodes
    file.close()

    # Create an empty list to store the RollList[min] values
    global roll_list_values 
    interface = cv2.imread("static/interface/newinterface.jpg")
    interface = cv2.resize(interface, (1240, 680))
    # bl = cv2.imread("interface/blackpage.png")
    # bl = cv2.resize(bl, (1230 - 800, 670 - 10))
    # interface[10:670, 800:1230] = bl


    # cv2.namedWindow(winname="interface")


    # interface = cv2.rectangle(interface, (127, 116), (640 + 127, 116 + 486), (0, 255, 50), 3)

   
    # ret, frame = cap.read()
    frame=cv2.imread("static/data/attendance.jpg")
    h, w, channels = frame.shape
    frame=cv2.resize(frame,(2*w,2*h))
    # ph=h
    # pw=w//2
    # parts = []
    # allFacesInParts=[]
    # encodeAllFacesInParts=[]
    # for i in range(2):
            
                # x_start = i * pw
                # x_end = (i + 1) * pw
                # y_start = 0
                # y_end =ph
                # part = frame[y_start:y_end, x_start:x_end]
                # he,we,_=part.shape
                # part=cv2.resize(part,(2*we,2*he))

                # part=cv2.resize(part,(640,486))
                # part=cv2.cvtColor(part,cv2.COLOR_BGR2RGB)
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    allFacesInParts=(face_recognition.face_locations(frame))
    encodeAllFacesInParts=face_recognition.face_encodings(frame,face_recognition.face_locations(frame))
                # parts.append(part)
    # part=frame[0:ph,w//3:2*w//3]
    # he,we,_=part.shape
    # part=cv2.resize(part,(2*we,2*he))
    # part=cv2.resize(part,(640,486))
    # part=cv2.cvtColor(part,cv2.COLOR_BGR2RGB)
    # allFacesInParts.append(face_recognition.face_locations(part))
    # encodeAllFacesInParts.append(face_recognition.face_encodings(part,face_recognition.face_locations(part)))
    # parts.append(part)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for faceloc in allFacesInParts:
                    frame=cv2.rectangle(frame,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,255,0),4)                        
        
    for encodes in encodeAllFacesInParts:
                facedis = face_recognition.face_distance(encodings, encodes)

                min_index=np.argmin(facedis)
                roll_value=[]
                if facedis[min_index]<0.55:
                    # print(facedis[min_index])
                    roll_value=RollList[min_index][:-2]
                    roll_list_values.append(roll_value)
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
        
    # for i in range(2):
    #     parts[i]=cv2.resize(parts[i],(pw,ph))
    # frame = np.zeros((h, w, 3), dtype=np.uint8)
    # part_index = 0
    # for i in range(2):
            
    #             x_start = i * pw
    #             x_end = (i + 1) * pw
    #             y_start = 0
    #             y_end =  ph

              
    #             frame[y_start:y_end, x_start:x_end] = parts[part_index]
    #             part_index += 1
    frame = cv2.resize(frame, (640, 486))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    interface[116:116 + 486, 127:640 + 127] = frame
        # interface=cv2.resize(interface,(400,400))
        # break
    # _, buffer = cv2.imencode('.png', interface)
    # attendances = base64.b64encode(buffer).decode()
    # attendancer=ndarray_to_b64(interface)
    # _, image_data = cv2.imencode('.png', interface)
    cv2.imwrite('static/interface.jpg',interface)
    # return render_template('interface.html')
    # cv2.imshow("interface", interface)
    # cv2.waitKey(0)
        # if cv2.waitKey(1) == 27 :
        #     break
    # print(roll_list_values)
    # After computing roll_list_values
    # session['roll_list_values'] = roll_list_values
    print(roll_list_values)

    return render_template('interface.html',subjects=subjects,roll_list_values=roll_list_values)

@app.route('/<subjects>/interfacevideo')
@login_required
def video(subjects):
    cap=cv2.VideoCapture(0)
    db=access_database(subjects)
    create_student_info_collection(db)
    today_data_attendance_collection=create_attendance_collection(db)
    attendance_all_collection = create_or_get_attendance_all_collection(db)
    cap = cv2.VideoCapture(0)

    file = open("encodings2.p", "rb")
    encodes = pickle.load(file)
    encodings, RollList = encodes
    file.close()

    # Create an empty list to store the RollList[min] values
    global roll_list_values 
    interface = cv2.imread("static/interface/newinterface.jpg")
    interface = cv2.resize(interface, (1240, 680))
    # bl = cv2.imread("interface/blackpage.png")
    # bl = cv2.resize(bl, (1230 - 800, 670 - 10))
    # interface[10:670, 800:1230] = bl


    # cv2.namedWindow(winname="interface")
    cv2.namedWindow("interface", cv2.WINDOW_NORMAL)


    # interface = cv2.rectangle(interface, (127, 116), (640 + 127, 116 + 486), (0, 255, 50), 3)

    while True:

        ret, frame = cap.read()
        # frame=cv2.imread("static/data/attendance.jpg")
        # h, w, channels = frame.shape
        # ph=h
        # pw=w//2
        # parts = []
        # allFacesInParts=[]
        # encodeAllFacesInParts=[]
        # for i in range(2):
            
        #         x_start = i * pw
        #         x_end = (i + 1) * pw
        #         y_start = 0
        #         y_end =ph
        #         part = frame[y_start:y_end, x_start:x_end]
        #         he,we,_=part.shape
        #         part=cv2.resize(part,(2*we,2*he))

        #         # part=cv2.resize(part,(640,486))
        #         part=cv2.cvtColor(part,cv2.COLOR_BGR2RGB)
        #         allFacesInParts.append(face_recognition.face_locations(part))
        #         encodeAllFacesInParts.append(face_recognition.face_encodings(part,face_recognition.face_locations(part)))
        #         parts.append(part)
        # part=frame[0:ph,w//3:2*w//3]
        # he,we,_=part.shape
        # part=cv2.resize(part,(2*we,2*he))
    # part=cv2.resize(part,(640,486))
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        allFacesInParts=(face_recognition.face_locations(frame))
        encodeAllFacesInParts=(face_recognition.face_encodings(frame,face_recognition.face_locations(frame)))
        for faceloc in allFacesInParts:
            frame=cv2.rectangle(frame,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,255,0),4)                        

        # parts.append(part)
        # for i in range(len(allFacesInParts)):
        for encodes in encodeAllFacesInParts:
                facedis = face_recognition.face_distance(encodings, encodes)

                min_index=np.argmin(facedis)
                if facedis[min_index]<0.5:
                    # print(facedis[min_index])
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
        #video
        # for i in range(2):
            # parts[i]=cv2.resize(parts[i],(pw,ph))
        # frame = np.zeros((h, w, 3), dtype=np.uint8)

        # part_index = 0
        # for i in range(2):
            
        #         x_start = i * pw
        #         x_end = (i + 1) * pw
        #         y_start = 0
        #         y_end =  ph

              
        #         frame[y_start:y_end, x_start:x_end] = parts[part_index]
        #         part_index += 1
        frame = cv2.resize(frame, (640, 486))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        interface[116:(116 + 486), 127:(640 + 127)] = frame
        # interface=cv2.resize(interface,(400,400))
        # break
    # _, buffer = cv2.imencode('.png', interface)
    # attendances = base64.b64encode(buffer).decode()
    # attendancer=ndarray_to_b64(interface)
    # _, image_data = cv2.imencode('.png', interface)
    # cv2.imwrite('static/interface.jpg',interface)
    # return render_template('interface.html')
        cv2.imshow("interface", interface)
    # cv2.waitKey(0)
        if cv2.waitKey(1) == 27 or cv2.getWindowProperty("interface", cv2.WND_PROP_VISIBLE) < 1:
            break
    # print(roll_list_values)
    # After computing roll_list_values
    # session['roll_list_values'] = roll_list_values
    print(roll_list_values)
    return render_template('biinterface.html',subjects=subjects)
    # return render_template('interface.html',subjects=subjects,roll_list_values=roll_list_values)
#docs
@app.route('/<subjects>/docs', methods=['POST', 'GET'])
@login_required
def docs(subjects):
    # roll_list_values = session.get('roll_list_values', [])
    # print(roll_list_values)
    global roll_list_values

    # roll_list_values=[]
    # for i in range(86):
    #     roll_list_values.append(i+1)

    for roll_number in roll_list_values:
        print("try",roll_number)

    roll_no_list = [0]*87
    for i in range(1, 87):
        if (len(str(i)) == 1):
            roll_no_list[i] =  int("22000100"+str(i))
        else :
            roll_no_list[i] = int("2200010"+str(i))

    total_attendence = []
    m = 0
    #--------------------------------- info1--------------------------------------------------------------------------------
    if 'input_date' in request.form:
        if request.method == 'POST':
            input_date = request.form['input_date']
            print(input_date)
            month = (str(input_date[5]) + str(input_date[6]))
            date = (str(input_date[8]) + str(input_date[9]))
            m = len(month)
            day = int((date) + (month))

            arr = [0]*87
            today = datetime.today()
            formatted_date = today.strftime("%Y-%m-%d")
            arr[0] = int(((str(formatted_date[8]) + str(formatted_date[9]))) + ((str(formatted_date[5]) + str(formatted_date[6]))))
            for x in roll_list_values:
                arr[int(x)%100] = 1

            with open('total_attendence.pkl','rb') as file:
                total_attendence=pickle.load(file)
            # total_attendence = []

            # print(arr)
            # print(roll_list_values)
            total_attendence.append(arr)

            with open('total_attendence.pkl', 'wb') as file:
                pickle.dump(total_attendence, file)
            # total_attendence = []
            print(total_attendence)

            l = ["Absent"] * 87
            for x in total_attendence:
                if x[0] == day:
                    for i in range(1, 87):
                        if x[i] == 1:
                            l[i] = "Present"

            # print(l)

            # for x in roll_list_values:
            #     l[(x)%100] = "Present"
            db=pd.DataFrame({
                "Roll_no":roll_no_list,
                "Attendance":l
            })
            db.to_csv("z.csv",index=False)
            path='./z.csv'
            return send_file(path,as_attachment=True)
        return render_template('docs.html',subjects=subjects,roll_list_values=roll_list_values)


    
    # ----------------------------------info2----------------------------------------------------
#   elif 'input_roll' in request.form:
    elif 'input_roll' in request.form:
            # Handle input_roll case
            if request.method == 'POST':
                input_roll = request.form['input_roll']
                # print(input_roll)

                with open('total_attendence.pkl','rb') as file:
                    total_attendence=pickle.load(file)

                # print(total_attendence)

                input_roll1 = int(input_roll)%100
                rollNo = []
                date_01 = []
                # for x in total_attendence:
                #     rollNo.append(x[input_roll1])
                #     date_01.append(x[0])
                for i in range(0, len(total_attendence)):
                    rollNo.append(total_attendence[i][input_roll1])
                    date_01.append(total_attendence[i][0])

                date_02 = []

                for x in date_01:
                    if len(str(x)) == 2:
                        date_02.append("2023-0"+str(x)[1]+"-0"+str(x)[0])
                    elif len(str(x)) == 3:
                        if m == 1 :
                            date_02.append("2023-0"+str(x)[2]+"-"+str(x)[0]+str(x)[1])
                        else :
                            date_02.append("2023-"+str(x)[1]+str(x)[2]+"-0"+str(x)[0])
                    else :
                        date_02.append("2023-"+str(x)[2]+str(x)[3]+"-"+str(x)[0]+str(x)[1])

                # print(rollNo)
                # print(date_01)

                db=pd.DataFrame({
                    "Date":date_02,
                    "Attendance":rollNo
                })
                db.to_csv("z.csv",index=False)
                path='./z.csv'
                return send_file(path,as_attachment=True)
            return render_template('docs.html',subjects=subjects)



    elif 'input_percent' in request.form:
            # Handle input_percent case
            if request.method == 'POST':
                input_percent = request.form['input_percent']
                print(input_percent)

                with open('total_attendence.pkl','rb') as file:
                    total_attendence=pickle.load(file)

                percent = [0]*87
                for i in range(1, 87):
                    sum = 0
                    for x in total_attendence:
                        if x[i] == 1:
                            sum += 1
                    percent[i] = (sum*100)//len(total_attendence)
                less_percent = []
                for i in range(1, 86):
                    if percent[i] <= int(input_percent):
                        if len(str(i)) == 1:
                            less_percent.append(int("22000100"+str(i)))
                        else :
                            less_percent.append(int("2200010"+str(i)))
                db=pd.DataFrame({
                    "Roll NO":less_percent
                    # "Attendance":rollNo
                })
                db.to_csv("z.csv",index=False)
                path='./z.csv'
                return send_file(path,as_attachment=True)
            return render_template('docs.html',subjects=subjects)
    
    return render_template('docs.html',subjects=subjects,roll_list_values=roll_list_values)


@app.route('/<subjects>/docsss', methods=['POST', 'GET'])
@login_required
def docs2(subjects):
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
# Handle input_percent case
    elif 'input_percents' in request.form:
        # Handle input_percent case
        if request.method == 'POST':
            input_percent = request.form['input_percents']
            print(input_percent)
            print(f'Number of "attendance_all" collections: {number_of_collections}')
            number_of_classes = float((float(input_percent) / 100) * number_of_collections)
            print(f'Number of classes based on {input_percent}%: {number_of_classes}')

            # Create a dictionary to count the appearances of roll_numbers
            roll_number_counts = defaultdict(int)

            # Ensure 'all_collections' is defined here
            all_collections = db.list_collection_names()

            # Iterate through all "attendance_all" collections
            for collection_name in all_collections:
                if collection_name.endswith("_attendance_all"):
                    collection = db[collection_name]

                    # Find all documents in the collection
                    all_students = collection.find({})

                    # Increment the count of each roll_number
                    for student in all_students:
                        roll_number = ''.join(student["roll_number"]) 
                        # roll_number = student["roll_number"]
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
            # Clear the global list before populating it again
            eligible_roll_numbers.clear()
            for roll_number, count in roll_number_counts.items():
                if count < number_of_classes:
                    eligible_roll_numbers.append(roll_number)

            # Print the eligible_roll_numbers in the console
            print("Eligible Roll Numbers:")
            print(eligible_roll_numbers)

        return redirect('new_route')


    return render_template('docs.html',subjects=subjects)

# @app.route('/<subjects>/docsss', methods=['POST', 'GET'])
# @login_required
# def docs2(subjects):
#     db=access_database(subjects)
#     # List all collection names in the database
#     all_collections = db.list_collection_names()

#     # Filter and count collections ending with "attendance_all"
#     attendance_collections = [collection for collection in all_collections if collection.endswith("_attendance_all")]
#     number_of_collections = len(attendance_collections)

#     # Print the number of "attendance_all" collections
#     print(f'Number of "attendance_all" collections: {number_of_collections}')

#      #--------------------------------- info1--------------------------------------------------------------------------------
#     if 'input_date' in request.form:
#         if request.method == 'POST':
#             input_date = request.form['input_date']

        
#         # Assuming your collection is named "{input_date}_attendance_all"
#             collection_name = f"{input_date}_attendance_all"

#             # Get the specified collection
#             collection = db[collection_name]

#             # Find all documents in the collection
#             result = list(collection.find())

#             if result:
#                 # Convert the result to a DataFrame
#                 df = pd.DataFrame(result)

#                 # Drop the first (index 0) and third (index 2) columns by integer position
#                 df = df.drop(df.columns[[0, 2]], axis=1)

#                 # Create a BytesIO object to save the Excel file
#                 excel_output = BytesIO()
#                 writer = pd.ExcelWriter(excel_output, engine='xlsxwriter')
#                 df.to_excel(writer, sheet_name='Sheet1', index=False)
#                 writer.close()  # Close the ExcelWriter

#                 excel_output.seek(0)

#                 # Send the Excel file as a downloadable attachment with the input_date variable as the filename
#                 return send_file(excel_output, as_attachment=True, download_name=f'{input_date}_{subjects}_attendance_all.xlsx')
                
#             else :
#                 print("The attendance is not present of that day")
#                 return render_template('noattendance.html')
    
#     # ----------------------------------info2----------------------------------------------------
# #   elif 'input_roll' in request.form:
#     elif 'input_roll' in request.form:
#             # Handle input_roll case
#             if request.method == 'POST':
#                 input_roll = request.form['input_roll']

#                 # Get all collection names that end with "_attendance_all"
#                 all_collections = db.list_collection_names()
#                 matching_dates = []  # To store dates where the student was present

#                 for collection_name in all_collections:
#                     if collection_name.endswith('_attendance_all'):
#                         collection = db[collection_name]
#                         if collection.find_one({"roll_number": input_roll, "remark": "Present"}):
#                             date = extract_date_from_collection(collection_name)
#                             matching_dates.append(date)

#                 if matching_dates:
#                     # Create a DataFrame to store the matching dates
#                     data = {'Dates': matching_dates}
#                     df = DataFrame(data)

#                     # Create an Excel file from the DataFrame
#                     excel_output = BytesIO()
#                     writer = pd.ExcelWriter(excel_output, engine='xlsxwriter')
#                     df.to_excel(writer, sheet_name='Sheet1', index=False)
#                     writer.close()
#                     excel_output.seek(0)

#                     # Return the Excel file as a downloadable attachment
#                     return send_file(excel_output, as_attachment=True, download_name=f'{input_roll}_{subjects}_attendance_dates.xlsx')

#                 else:
#                     return render_template('noattendance.html')
#     elif 'input_percents' in request.form:
#             # Handle input_percent case
#             if request.method == 'POST':
#                 input_percent = request.form['input_percents']
#                 print(input_percent) 
#                 print(f'Number of "attendance_all" collections: {number_of_collections}')
#                 number_of_classes = float((float(input_percent) / 100) * number_of_collections)
#                 print(f'Number of classes based on {input_percent}%: {number_of_classes}')
#                 # Create a dictionary to count the appearances of roll_numbers
#                 roll_number_counts = defaultdict(int)

#                 # Ensure 'all_collections' is defined here
#                 all_collections = db.list_collection_names()

#                 # Create a dictionary to count the appearances of roll_numbers
#                 roll_number_counts = defaultdict(int)

#                 # Iterate through all "attendance_all" collections
#                 for collection_name in all_collections:
#                     if collection_name.endswith("_attendance_all"):
#                         collection = db[collection_name]

#                         # Find all documents in the collection
#                         all_students = collection.find({})

#                         # Increment the count of each roll_number
#                         for student in all_students:
#                             roll_number = student["roll_number"]
#                             print(f"Roll Number type: {type(roll_number)}")
#                             roll_number_remark = student.get("remark", "")  # Use get to handle cases where "remark" is missing

#                             if roll_number_remark == "Present":
#                                 roll_number_counts[roll_number] += 1

#                             elif roll_number_remark == "Absent":
#                                 roll_number_counts[roll_number] += 0

#                 print("Roll Number Counts:")
#                 for roll_number, count in roll_number_counts.items():
#                     print(f"Roll Number: {roll_number}, Count: {count}")

#                 # Print the counts for roll numbers that have counts less than number_of_classes
#                 print("Roll Numbers with Counts Less than Number of Classes:")
#                 for roll_number, count in roll_number_counts.items():
#                     if count < number_of_classes:
#                         print(f"Roll Number: {roll_number}, Count: {count}")

#                 # Filter roll_numbers with counts less than number_of_classes
#                 # Create a list to store eligible roll numbers
#                 global eligible_roll_numbers
#                 eligible_roll_numbers = [roll_number for roll_number, count in roll_number_counts.items() if count < number_of_classes]

#                 # Print the eligible_roll_numbers in the console
#                 print("Eligible Roll Numbers:")
#                 print(eligible_roll_numbers)
                
#             return redirect('new_route')

#     return render_template('docs.html',subjects=subjects)

    # cap.release()
    # cv2.destroyAllWindows()
    return render_template('index.html',subjects=subjects)

# Define the folder where you'll store uploaded images.
UPLOAD_FOLDER = './static/data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

image=[]
@app.route('/<subjects>/upload', methods=['GET', 'POST'])
@login_required
def upload_file(subjects):
    if request.method == 'POST':
        if 'image' in request.files:
            uploaded_image = request.files['image']
            if uploaded_image.filename != '':
                # Save the uploaded image with a fixed name
                uploaded_image.save(os.path.join(app.config['UPLOAD_FOLDER'], 'attendance.jpg'))
                return "Image uploaded successfully as 'attendance.jpg'"

    return render_template('upload.html',subjects=subjects)


@app.route('/<subjects>/uploads/attendance.jpg')
@login_required
def uploaded_file(subjects):
    return send_from_directory(app.config['UPLOAD_FOLDER'], 'attendance.jpg')





# if __name__ == "__main__":
#     app.run(debug=True,port=3000)
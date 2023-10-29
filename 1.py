from datetime import datetime
from pymongo import MongoClient

def generate_roll_numberlist():
    # Define the starting and ending roll numbers
    start_roll_number = 220001001
    end_roll_number = 220001082

    # List of roll numbers in the specified range
    roll_number_list = [str(roll) for roll in range(start_roll_number, end_roll_number + 1)]

    # Add the extra roll numbers
    extra_roll_numbers = ["220002018", "220002028", "220002063", "220002081"]
    roll_number_list.extend(extra_roll_numbers)
    return roll_number_list

def access_database(subject_name):
    client = MongoClient('mongodb://localhost:27017/') 
    db = client[subject_name]
    return db

def create_attendance_all_collection(db):
    # Create a new collection name for attendance_all
    roll_number_list = generate_roll_numberlist()
    # today_date = today_date()
    today_date = datetime.now().strftime("%Y-%m-%d")
    attendance_all_collection_name = today_date + "_attendance_all"
    attendance_all_collection = db[attendance_all_collection_name]
    print(attendance_all_collection)  

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

# Access the database and create the attendance collection
db = access_database('trim')
create_attendance_all_collection(db)

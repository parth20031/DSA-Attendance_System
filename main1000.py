# -----------


# @app.route('/<subjects>/docs', methods=['POST', 'GET'])
# @login_required
# def docs(subjects):
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
#     elif 'input_percent' in request.form:
#             # Handle input_percent case
#             if request.method == 'POST':
#                 input_percent = request.form['input_percent']
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

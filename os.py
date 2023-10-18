import os
files=os.listdir("student_details")
# print (files)
for file in files:
    filename, file_extension = os.path.splitext(file)
    old = os.path.join("student_details",file)
    newname=f"{filename}1{file_extension}"
    new=os.path.join("student_details",newname)
    # print(new)
    os.rename(old,new)
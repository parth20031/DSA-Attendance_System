from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Use a list to store the subjects (you should use a database in a production environment)
subjects = []

@app.route('/')
def home():
    return render_template('index2.html', subjects=subjects)

@app.route('/add_subject', methods=['POST'])
def add_subject():
    subject_name = request.form.get('subject_name')
    if subject_name:
        subjects.append(subject_name)
    return redirect(url_for('home'))

@app.route('/start')
def start():
    

if __name__ == "__main__":
    app.run(debug=True)

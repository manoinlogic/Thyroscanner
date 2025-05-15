import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from flask import Flask, request, render_template
import contextlib
import joblib
import re
import sqlite3
import pandas as pd
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from create_database import setup_database
from tensorflow.keras.applications.vgg16 import preprocess_input
from utils import login_required, set_session
from flask import (
    Flask, render_template, 
    request, session, redirect
)

# Initialize Flask app
app = Flask(__name__)

# Load trained model
MODEL_PATH = 'model/optimized_model.h5'
model = load_model(MODEL_PATH)

database = "users.db"
setup_database(name=database)

app.secret_key = 'xpSm7p5bgJY8rNoBjGWiz5yjxM-NEBlW6SIBI62OkLc='

# Define class labels (Ensure this matches your training dataset)
class_labels = ['abnormal','normal']  # Update this based on your dataset

# Ensure upload folder exists
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diet')
def diet():
    return render_template('diet.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')

    # Set data to variables
    username = request.form.get('username')
    password = request.form.get('password')
    
    # Attempt to query associated user data
    query = 'select username, password, email from users where username = :username'

    with contextlib.closing(sqlite3.connect(database)) as conn:
        with conn:
            account = conn.execute(query, {'username': username}).fetchone()

    if not account: 
        return render_template('login.html', error='Username does not exist')

    # Verify password
    try:
        ph = PasswordHasher()
        ph.verify(account[1], password)
    except VerifyMismatchError:
        return render_template('login.html', error='Incorrect password')

    # Check if password hash needs to be updated
    if ph.check_needs_rehash(account[1]):
        query = 'update set password = :password where username = :username'
        params = {'password': ph.hash(password), 'username': account[0]}

        with contextlib.closing(sqlite3.connect(database)) as conn:
            with conn:
                conn.execute(query, params)

    # Set cookie for user session
    set_session(
        username=account[0], 
        email=account[2], 
        remember_me='remember-me' in request.form
    )
    
    return redirect('/predict_page')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    
    # Store data to variables 
    password = request.form.get('password')
    confirm_password = request.form.get('confirm-password')
    username = request.form.get('username')
    email = request.form.get('email')

    # Verify data
    if len(password) < 8:
        return render_template('register.html', error='Your password must be 8 or more characters')
    if password != confirm_password:
        return render_template('register.html', error='Passwords do not match')
    if not re.match(r'^[a-zA-Z0-9]+$', username):
        return render_template('register.html', error='Username must only be letters and numbers')
    if not 3 < len(username) < 26:
        return render_template('register.html', error='Username must be between 4 and 25 characters')

    query = 'select username from users where username = :username;'
    with contextlib.closing(sqlite3.connect(database)) as conn:
        with conn:
            result = conn.execute(query, {'username': username}).fetchone()
    if result:
        return render_template('register.html', error='Username already exists')

    # Create password hash
    pw = PasswordHasher()
    hashed_password = pw.hash(password)

    query = 'insert into users(username, password, email) values (:username, :password, :email);'
    params = {
        'username': username,
        'password': hashed_password,
        'email': email
    }

    with contextlib.closing(sqlite3.connect(database)) as conn:
        with conn:
            result = conn.execute(query, params)

    # We can log the user in right away since no email verification
    set_session( username=username, email=email)
    return redirect('/')

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')

def predict_ovarian_cancer(image_path):
    """Preprocess the image and make a prediction"""
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Get class index
    confidence = np.max(predictions) * 100  # Get confidence percentage

    return class_labels[predicted_class], confidence

@app.route("/predict_page", methods=["POST"])
def predict():
    """Handle image upload and prediction"""
    if "image" not in request.files:
        return render_template("index.html", error="No image uploaded")

    file = request.files["image"]
    
    if file.filename == "":
        return render_template("index.html", error="No selected file")

    # Save uploaded file
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Get prediction
    predicted_label, confidence = predict_ovarian_cancer(file_path)

    return render_template("result.html", image=file.filename, predicted_class=predicted_label, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, Response, flash, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, current_user, logout_user, login_required
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Length, EqualTo,ValidationError,InputRequired
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3
import keyboard
from flask_bcrypt import Bcrypt

app = Flask(__name__)


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
string = " "

classifier = Classifier("model/keras_model.h5", "model/labels.txt")
labels = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J", 10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 19: "T", 20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z"}

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'justpraveen'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
def generate_frames():

    global string
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            success, frame = cap.read()
            img1 = frame.copy()
            hands, frame = detector.findHands(img1)
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = frame[y - offset:y + h + offset, x - offset:x + w + offset]
                imgCropShape = imgCrop.shape
                aspectRatio = h / w
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    if (x > 0 + offset and y > 0 + offset and w > 0 + offset and h > 0 + offset):
                        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgSize - wCal) / 2)
                        imgWhite[:, wGap:wCal + wGap] = imgResize
                        prediction, index = classifier.getPrediction(imgWhite, draw=False)
                        print(prediction, index, labels[index])
                        if keyboard.is_pressed('enter'):
                            string += labels[index]
                        if keyboard.is_pressed('spacebar'):
                            string += " "
                        if keyboard.is_pressed('backspace'):
                            string = string[:-1]
                        if keyboard.is_pressed('delete'):
                            string = ""
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    if (x > 0 + offset and y > 0 + offset and w > 0 + offset and h > 0 + offset):
                        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgSize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize
                        prediction, index = classifier.getPrediction(imgWhite, draw=False)
                        print(prediction, index, labels[index])
                        if keyboard.is_pressed('enter'):
                            string += labels[index]
                        if keyboard.is_pressed('spacebar'):
                            string += " "
                        if keyboard.is_pressed('backspace'):
                            string = string[:-1]
                        if keyboard.is_pressed('delete'):
                            string = ""
                cv2.rectangle(img1, (x - 50, y - offset - 50),
                              (x + 10 + 90, y - offset - 50 + 50), (255, 0, 0), cv2.FILLED)
                cv2.putText(img1, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                cv2.rectangle(img1, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (255, 0, 255), 4)
                cv2.putText(img1, string, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            ret, buffer = cv2.imencode('.jpg', img1)
            img1 = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img1 + b'\r\n')

    return render_template("index.html", string=string)




@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))



# User model
class User(db.Model,UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)
    saved_strings = db.relationship('SavedString', backref='user', lazy=True)


class SavedString(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    string = db.Column(db.String(255), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)


# Registration form
class RegistrationForm(FlaskForm):
    username = StringField( validators=[InputRequired(), Length(min=4, max=20)],render_kw={"placeholder":"Username"})
    password = PasswordField( validators=[InputRequired(), Length(min=6, max=20)],render_kw={"placeholder":"Password"})
    submit = SubmitField('Sign Up')
    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            flash("Username already taken!","warning")
            raise ValidationError(
                'That username already exists. Please choose a different one.')


# Login form
class LoginForm(FlaskForm):
    username = StringField( validators=[InputRequired(), Length(min=4, max=20)],render_kw={"placeholder":"Username"})
    password = PasswordField( validators=[InputRequired(), Length(min=6, max=20)],render_kw={"placeholder":"Password"})
    submit = SubmitField('Log In')



def initialize_camera():
    global cap
    cap = cv2.VideoCapture(0)
@ app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful. You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html', form=form)




@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                initialize_camera()


                return redirect(url_for('index'))
        flash('Invalid username or password', 'danger')
    return render_template('login.html', form=form)



@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    global string
    string =""
    cap.release()

    logout_user()

    return redirect(url_for('login'))





@app.route('/stop', methods=['POST', 'GET'])
@login_required
def stopping():
    new_saved_string = SavedString(string=string,user=current_user)
    db.session.add(new_saved_string)
    db.session.commit()
    engine = pyttsx3.init()
    newVoiceRate = 125
    engine.setProperty('rate', newVoiceRate)
    engine.say(string)
    engine.runAndWait()
    return render_template('index.html', string=string)


@app.route('/get_string')
def get_string():
    return string



@app.route('/video')
@login_required
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST', 'GET'])
@login_required
def predictions():
    return render_template("index.html", string=string)
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/saved_strings')
@login_required
def saved_strings():
    #saved_strings = SavedString.query.all()
    saved_strings = current_user.saved_strings
    return render_template('saved_strings.html', saved_strings=saved_strings)
@app.route('/index')
@login_required
def index():
    saved_strings= current_user.saved_strings
    return render_template('index.html', string=string)
if __name__ == "__main__":
    app.run(debug=True)



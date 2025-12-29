from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# Initialize the database object
db = SQLAlchemy()

# Define the User Table as a Python Class
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(50), nullable=False)  # 'doctor' or 'patient'
    
    # Relationship: One user has many history records
    history = db.relationship('History', backref='patient', lazy=True)

# Define the History Table
class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.String(50), default=datetime.now().strftime("%Y-%m-%d %H:%M"))
    age = db.Column(db.Integer)
    bp = db.Column(db.Integer)
    cholesterol = db.Column(db.Integer)
    risk_level = db.Column(db.String(100))
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime


db = SQLAlchemy()


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class AIAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    input_text = db.Column(db.Text, nullable=False)
    ai_probability = db.Column(db.Float, nullable=False)
    conclusion = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class SimilarityAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    source_text = db.Column(db.Text, nullable=False)
    comparison_text = db.Column(db.Text, nullable=False)
    similarity_percentage = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class PlagiarismAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    source_text = db.Column(db.Text, nullable=False)
    suspicious_text = db.Column(db.Text, nullable=False)
    similarity_score = db.Column(db.Float, nullable=False)
    plagiarism_level = db.Column(db.String(20), nullable=False)
    message = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class WebPlagiarismAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey(User.id), nullable=False)
    checked_text = db.Column(db.Text, nullable=False)
    source_url = db.Column(db.String(500), nullable=False)
    source_title = db.Column(db.String(255), nullable=True)
    similarity_score = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

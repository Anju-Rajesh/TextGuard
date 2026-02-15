import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from models import db, User, AIAnalysis, SimilarityAnalysis, PlagiarismAnalysis, WebPlagiarismAnalysis
from utils.ai_detector import analyze_text_ai
from utils.similarity_checker import calculate_similarity, get_plagiarism_level
from utils.plagiarism_detector import detect_plagiarism_from_corpus

# --- APP CONFIGURATION ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here' # Used for signing session cookies
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db' # Use SQLite file database
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- INITIALIZE EXTENSIONS ---
db.init_app(app)       # Connect database
bcrypt = Bcrypt(app)   # Initialize password hashing tool
login_manager = LoginManager(app) # Initialize login session manager
login_manager.login_view = 'login' # Redirect here if user is not logged in

@login_manager.user_loader
def load_user(user_id):
    """Reloads the user object from the user ID stored in the session."""
    return db.session.get(User, int(user_id))

# Create database tables if they don't exist
with app.app_context():
    db.create_all()

# --- ROUTES (URL HANDLING) ---

@app.route('/')
def index():
    return redirect(url_for('ai_detection'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handles User Registration: Hashes password and saves to DB."""
    if current_user.is_authenticated:
        return redirect(url_for('ai_detection'))
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        user_exists = User.query.filter((User.username == username) | (User.email == email)).first()
        if user_exists:
            flash('Username or email already exists.', 'danger')
            return redirect(url_for('register'))
        
        # Security: Hash the password before saving
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, email=email, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handles User Login: Checks password hash."""
    if current_user.is_authenticated:
        return redirect(url_for('ai_detection'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        
        # Verify password against hash
        if user and bcrypt.check_password_hash(user.password_hash, password):
            login_user(user) # Create session
            return redirect(url_for('ai_detection'))
        else:
            flash('Login unsuccessful. Please check email and password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/ai-detection', methods=['GET', 'POST'])
@login_required # Protects route: Login required to access
def ai_detection():
    """Main Feature: Analyzes text style to detect AI generation."""
    result = None
    if request.method == 'POST':
        text = ""
        # Handle file upload (.txt) or direct text paste
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            if file.filename.endswith('.txt'):
                text = file.read().decode('utf-8')
            else:
                flash('Only .txt files are allowed.', 'warning')
        else:
            text = request.form.get('text', '')

        if text.strip():
            # Run the AI Detection Logic (from utils/ai_detector.py)
            probability, conclusion = analyze_text_ai(text)
            
            # Save results to database linked to current user
            analysis = AIAnalysis(user_id=current_user.id, input_text=text[:500] + '...', ai_probability=probability, conclusion=conclusion)
            db.session.add(analysis)
            db.session.commit()
            
            result = {'probability': probability, 'conclusion': conclusion}
        else:
            flash('Please provide some text or upload a file.', 'info')
            
    return render_template('ai_detection.html', result=result)

@app.route('/similarity', methods=['GET', 'POST'])
@login_required
def similarity():
    """Main Feature: Checks similarity between two texts (Plagiarism)."""
    result = None
    if request.method == 'POST':
        source_text = request.form.get('source_text', '')
        comparison_text = request.form.get('comparison_text', '')
        
        if source_text.strip() and comparison_text.strip():
            # Run Similarity Logic (TF-IDF Cosine Similarity)
            sim_percentage = calculate_similarity(source_text, comparison_text)
            
            # Save to database
            analysis = SimilarityAnalysis(
                user_id=current_user.id, 
                source_text=source_text[:500] + '...', 
                comparison_text=comparison_text[:500] + '...', 
                similarity_percentage=sim_percentage
            )
            db.session.add(analysis)
            db.session.commit()
            
            result = {'percentage': sim_percentage}
        else:
            flash('Please provide both source and comparison text.', 'info')
            
    return render_template('similarity.html', result=result)


@app.route('/plagiarism', methods=['GET', 'POST'])
@login_required
def plagiarism():
    """Main Feature: Checks for plagiarism using Local Corpus."""
    corpus_results = None
    
    if request.method == 'POST':
        action = request.form.get('action')

        # ---------------- CORPUS CHECK ----------------
        if action == 'corpus_check':
            text_to_check = ""

            if 'corpus_file' in request.files and request.files['corpus_file'].filename != '':
                file = request.files['corpus_file']
                if file.filename.endswith('.txt'):
                    text_to_check = file.read().decode('utf-8')
                else:
                    flash('Only .txt files are allowed.', 'warning')
            else:
                text_to_check = request.form.get('corpus_text', '')

            if text_to_check.strip():
                corpus_detection_result = detect_plagiarism_from_corpus(text_to_check)

                top_source = (
                    corpus_detection_result['top_sources'][0]['source']
                    if corpus_detection_result['top_sources']
                    else "No Match"
                )

                analysis = PlagiarismAnalysis(
                    user_id=current_user.id,
                    source_text=f"Corpus Search (Top: {top_source})",
                    suspicious_text=text_to_check[:500] + '...',
                    similarity_score=corpus_detection_result['overall_plagiarism_percentage'],
                    plagiarism_level=corpus_detection_result['plagiarism_level'],
                    message=corpus_detection_result['message']
                )
                db.session.add(analysis)
                db.session.commit()

                corpus_results = corpus_detection_result
            else:
                flash('Please provide text or file to check against corpus.', 'info')

    return render_template(
        'plagiarism.html',
        corpus_results=corpus_results
    )


if __name__ == '__main__':
    app.run(debug=True)





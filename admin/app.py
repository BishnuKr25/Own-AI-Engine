"""Production Flask Admin Portal"""
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SelectField, SubmitField
from wtforms.validators import DataRequired, Email, Length
from werkzeug.security import generate_password_hash, check_password_hash
import secrets
from datetime import datetime, timedelta
from pymongo import MongoClient
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', secrets.token_hex(32))
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)

# MongoDB connection
client = MongoClient(os.getenv('MONGODB_URL', 'mongodb://localhost:27017'))
db = client.sovereign_ai

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.username = user_data['username']
        self.email = user_data['email']
        self.role = user_data['role']

@login_manager.user_loader
def load_user(user_id):
    from bson import ObjectId
    user_data = db.admins.find_one({'_id': ObjectId(user_id)})
    if user_data:
        return User(user_data)
    return None

# Forms
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class CreateUserForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=50)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    organization = StringField('Organization', validators=[Length(max=100)])
    role = SelectField('Role', choices=[
        ('user', 'User'),
        ('developer', 'Developer'),
        ('admin', 'Admin')
    ])
    submit = SubmitField('Create User')

@app.route('/')
@login_required
def dashboard():
    """Admin dashboard"""
    # Get statistics
    stats = {
        'total_users': db.users.count_documents({}),
        'active_keys': db.api_keys.count_documents({'is_active': True}),
        'total_queries': db.query_logs.count_documents({}),
        'queries_today': db.query_logs.count_documents({
            'timestamp': {'$gte': datetime.utcnow().replace(hour=0, minute=0, second=0)}
        })
    }
    
    # Get recent activity
    recent_queries = list(db.query_logs.find().sort('timestamp', -1).limit(10))
    
    return render_template('dashboard.html', 
                         stats=stats,
                         recent_queries=recent_queries,
                         user=current_user)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Admin login"""
    form = LoginForm()
    
    if form.validate_on_submit():
        admin = db.admins.find_one({'username': form.username.data})
        
        if admin and check_password_hash(admin['password'], form.password.data):
            user = User(admin)
            login_user(user, remember=True)
            
            # Log login
            db.admin_logs.insert_one({
                'action': 'login',
                'admin': form.username.data,
                'timestamp': datetime.utcnow()
            })
            
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        
        flash('Invalid username or password', 'danger')
    
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    """Admin logout"""
    # Log logout
    db.admin_logs.insert_one({
        'action': 'logout',
        'admin': current_user.username,
        'timestamp': datetime.utcnow()
    })
    
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/users')
@login_required
def users():
    """User management"""
    # Get all users with their API keys
    users_list = []
    for user in db.users.find():
        api_keys = list(db.api_keys.find({'user_id': str(user['_id'])}))
        user['api_keys'] = api_keys
        users_list.append(user)
    
    return render_template('users.html', users=users_list, user=current_user)

@app.route('/create_user', methods=['GET', 'POST'])
@login_required
def create_user():
    """Create new user"""
    form = CreateUserForm()
    
    if form.validate_on_submit():
        # Check if username exists
        if db.users.find_one({'username': form.username.data}):
            flash('Username already exists', 'danger')
            return redirect(url_for('create_user'))
        
        # Check if email exists
        if db.users.find_one({'email': form.email.data}):
            flash('Email already registered', 'danger')
            return redirect(url_for('create_user'))
        
        # Create user
        user_data = {
            'username': form.username.data,
            'email': form.email.data,
            'organization': form.organization.data,
            'role': form.role.data,
            'created_at': datetime.utcnow(),
            'created_by': current_user.username,
            'is_active': True
        }
        
        result = db.users.insert_one(user_data)
        user_id = str(result.inserted_id)
        
        # Generate API key
        api_key = f"sk-{secrets.token_hex(32)}"
        api_key_hash = secrets.token_hex(32)  # In production, use proper hashing
        
        db.api_keys.insert_one({
            'key': api_key_hash,
            'display_key': f"sk-...{api_key[-8:]}",
            'full_key': api_key,  # Store temporarily for display
            'user_id': user_id,
            'created_at': datetime.utcnow(),
            'created_by': current_user.username,
            'is_active': True
        })
        
        # Log action
        db.admin_logs.insert_one({
            'action': 'create_user',
            'admin': current_user.username,
            'target_user': form.username.data,
            'timestamp': datetime.utcnow()
        })
        
        flash(f'User created successfully. API Key: {api_key}', 'success')
        return redirect(url_for('users'))
    
    return render_template('create_user.html', form=form, user=current_user)

# Initialize default admin
def init_admin():
    """Create default admin if none exists"""
    if db.admins.count_documents({}) == 0:
        admin_data = {
            'username': 'admin',
            'email': 'admin@sovereign-ai.local',
            'password': generate_password_hash('changeme123'),
            'role': 'super_admin',
            'created_at': datetime.utcnow()
        }
        db.admins.insert_one(admin_data)
        print("Default admin created - Username: admin, Password: changeme123")
        print("IMPORTANT: Change this password immediately!")

if __name__ == '__main__':
    init_admin()
    app.run(host='0.0.0.0', port=5000, debug=False)
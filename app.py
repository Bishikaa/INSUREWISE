from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from chatbot.chatbot import get_chatbot_response
from werkzeug.security import generate_password_hash, check_password_hash
import markdown
import re  
import sqlite3
import time
from functools import wraps

app = Flask(__name__)
app.secret_key = "insurewise123"

DB_NAME = 'users.db'

def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        email TEXT NOT NULL UNIQUE,
                        password TEXT NOT NULL
                    )''')
        # Add chat history table
        c.execute('''CREATE TABLE IF NOT EXISTS chat_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER NOT NULL,
                        user_query TEXT NOT NULL,
                        ai_response TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(user_id) REFERENCES users(id)
                    )''')
        conn.commit()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            if request.method == 'POST':
                return jsonify({'error': 'Please log in to use the chatbot.'}), 401
            flash("Please log in to use the chatbot.")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/faqs')
def faqs():
    return render_template('faqs.html') 


@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            c.execute("SELECT id, name, password FROM users WHERE email = ?", (email,))
            user = c.fetchone()

            if user and check_password_hash(user[2], password):
                session['user_id'] = user[0]
                session['user_name'] = user[1]
                flash("Logged in successfully!")
                return redirect(url_for('index'))
            else:
                flash("Invalid email or password.")
                return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully.")
    return redirect(url_for('index'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Email validation
        if not re.match(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$', email):
            flash("Invalid email format. Please enter a valid email address.")
            return render_template('signup.html', name=name, email=email)

        # Password validation
        if password != confirm_password:
            flash("Passwords don't match. Please make sure both password fields are the same.")
            return render_template('signup.html', name=name, email=email)

        # Password strength check
        if not re.match(r'^[A-Z][A-Za-z0-9]*[0-9]+.*$', password):
            flash("Password must start with an uppercase letter and contain at least one number.")
            return render_template('signup.html', name=name, email=email)

        try:
            hashed_password = generate_password_hash(password)
            with sqlite3.connect(DB_NAME) as conn:
                c = conn.cursor()
                c.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                          (name, email, hashed_password))
                conn.commit()
                flash("Account created successfully! Please log in.")
                return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Email already exists. Try logging in.")
            return render_template('signup.html', name=name, email=email)

    return render_template('signup.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if 'user_id' not in session:
        if request.method == 'POST':
            return jsonify({'error': 'Please log in to use the chatbot.'}), 401
        flash("Please log in to use the chatbot.") 
        return redirect(url_for('login'))

    if request.method == 'POST':
        user_message = request.form.get('user_message')
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
            
        try:
            # Get chat history for context
            history = get_chat_history(session['user_id'])
            
            # Get bot response (modified to accept history)
            bot_reply = get_chatbot_response(user_message, history)
            
            # Save both messages to database
            save_chat_message(session['user_id'], user_message, bot_reply)
            
            return bot_reply
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return jsonify({'error': 'Failed to generate response'}), 500

    # For GET requests, render the chat template with history
    messages = get_chat_history(session['user_id'])
    return render_template('chat.html', messages=messages)

def save_chat_message(user_id, user_query, ai_response):
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute("INSERT INTO chat_history (user_id, user_query, ai_response) VALUES (?, ?, ?)",
                  (user_id, user_query, ai_response))
        conn.commit()

def get_chat_history(user_id, limit=20):
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute("SELECT user_query, ai_response FROM chat_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?", 
                  (user_id, limit))
        history = [{'user_query': row[0], 'ai_response': row[1]} for row in c.fetchall()]
        return history[::-1]  # Reverse to show oldest first
# ... [previous code remains the same until the chat history functions]

def save_chat_message(user_id, user_query, ai_response):
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute("INSERT INTO chat_history (user_id, user_query, ai_response) VALUES (?, ?, ?)",
                  (user_id, user_query, ai_response))
        conn.commit()

def get_chat_history(user_id, limit=20):
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute("SELECT user_query, ai_response FROM chat_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?", 
                  (user_id, limit))
        history = [{'user_query': row[0], 'ai_response': row[1]} for row in c.fetchall()]
        return history[::-1]  # Reverse to show oldest first

# Add these two new routes here:
@app.route('/get_chat_history')
@login_required
def get_chat_history_endpoint():
    history = get_chat_history(session['user_id'])
    return jsonify(history)

@app.route('/clear_history', methods=['POST'])
@login_required
def clear_history():
    try:
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            c.execute("DELETE FROM chat_history WHERE user_id = ?", (session['user_id'],))
            conn.commit()
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# This should remain as the last block in the file
if __name__ == '__main__':
    init_db()
    app.run(debug=True, host="0.0.0.0", port=4000)


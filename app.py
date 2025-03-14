from flask import Flask, render_template, request, jsonify
from flask_mysqldb import MySQL

app = Flask(__name__)

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'your_username'
app.config['MYSQL_PASSWORD'] = 'your_password'
app.config['MYSQL_DB'] = 'your_database_name'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)

# Create users table if not exists
def create_users_table():
    cur = mysql.connection.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS users (
                   id INT AUTO_INCREMENT PRIMARY KEY,
                   username VARCHAR(50) NOT NULL,
                   email VARCHAR(100) NOT NULL
                   )''')
    mysql.connection.commit()
    cur.close()

create_users_table()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_user', methods=['POST'])
def add_user():
    username = request.form['username']
    email = request.form['email']
    
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO users (username, email) VALUES (%s, %s)", (username, email))
    mysql.connection.commit()
    cur.close()
    
    return 'User added successfully'

@app.route('/users')
def users():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users")
    users = cur.fetchall()
    cur.close()
    
    return jsonify(users)

if __name__ == '__main__':
    app.run(debug=True)

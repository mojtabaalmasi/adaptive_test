from flask import Flask, render_template, request, redirect, url_for
import os
import sqlite3

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/videos'
app.secret_key = 'secret123'

# ایجاد جدول‌ها در صورت عدم وجود
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            filename TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS subtitles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            time REAL,
            word TEXT
        )
    ''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS pre_test_questionnaire (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        participant_id INTEGER,
        native_speaker BOOLEAN,
        university TEXT,
        field_of_study TEXT,
        education_level TEXT,
        samfa_taken BOOLEAN,
        samfa_score REAL,
        vocab_writing TEXT,
        vocab_speaking TEXT,
        vocab_reading TEXT,
        vocab_listening TEXT,
        FOREIGN KEY (participant_id) REFERENCES participants(id)
        )
    ''')
    conn.commit()
    conn.close()

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        title = request.form['title']
        video = request.files['video']
        timestamps = request.form.getlist('timestamp')
        words = request.form.getlist('word')

        if video:
            filename = video.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video.save(filepath)

            conn = sqlite3.connect('database.db')
            c = conn.cursor()
            c.execute("INSERT INTO videos (title, filename) VALUES (?, ?)", (title, filename))
            video_id = c.lastrowid

            for t, w in zip(timestamps, words):
                if t.strip() and w.strip():
                    c.execute("INSERT INTO subtitles (video_id, time, word) VALUES (?, ?, ?)",
                              (video_id, float(t.strip()), w.strip()))

            conn.commit()
            conn.close()

            return redirect(url_for('upload'))

    return render_template('upload.html')

if __name__ == '__main__':
    init_db()
    app.run(debug=True)

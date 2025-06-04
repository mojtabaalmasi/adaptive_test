import sqlite3
from flask import Flask, render_template, request, redirect, url_for, g

app = Flask(__name__)
DATABASE = 'questions.db'

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

# صفحه ثبت نام شرکت کننده
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        full_name = request.form['full_name']
        age = int(request.form['age'])
        language = request.form['language']
        db = get_db()
        cursor = db.execute('INSERT INTO participants (full_name, age, language) VALUES (?, ?, ?)',
                            (full_name, age, language))
        db.commit()
        participant_id = cursor.lastrowid
        return redirect(url_for('test_question', participant_id=participant_id, question_number=1))
    return render_template('index.html')

# صفحه سوال آزمون - دریافت سوال بر اساس شماره و پارامترهای IRT (مثال ساده: بر اساس شماره سوال)
@app.route('/test/<int:participant_id>/<int:question_number>', methods=['GET', 'POST'])
def test_question(participant_id, question_number):
    db = get_db()

    if request.method == 'POST':
        selected_option = int(request.form['option'])
        # گرفتن سوال قبلی
        question = db.execute('SELECT * FROM questions WHERE id = ?', (question_number,)).fetchone()
        if question is None:
            return "سؤال پیدا نشد", 404

        # فرض کنید جواب درست گزینه 1 است برای مثال، تو پروژه‌ات باید درست ذخیره شده باشد
        correct_option = 1  
        correct = 1 if selected_option == correct_option else 0

        db.execute('INSERT INTO answers (participant_id, question_id, selected_option, correct) VALUES (?, ?, ?, ?)',
                   (participant_id, question_number, selected_option, correct))
        db.commit()

        # سوال بعدی را بارگذاری کن
        next_question_number = question_number + 1
        next_question = db.execute('SELECT * FROM questions WHERE id = ?', (next_question_number,)).fetchone()
        if next_question:
            return redirect(url_for('test_question', participant_id=participant_id, question_number=next_question_number))
        else:
            return redirect(url_for('test_result', participant_id=participant_id))

    # GET: بارگذاری سوال
    question = db.execute('SELECT * FROM questions WHERE id = ?', (question_number,)).fetchone()
    if question is None:
        return "سؤال پیدا نشد", 404

    question_dict = {
        'id': question['id'],
        'text': question['text'],
        'options': [question['option1'], question['option2'], question['option3'], question['option4']]
    }
    return render_template('test_question.html', question=question_dict, participant_id=participant_id)

# صفحه نتایج آزمون
@app.route('/result/<int:participant_id>')
def test_result(participant_id):
    db = get_db()
    participant = db.execute('SELECT * FROM participants WHERE id = ?', (participant_id,)).fetchone()
    answers = db.execute('SELECT * FROM answers WHERE participant_id = ?', (participant_id,)).fetchall()

    correct_count = sum(a['correct'] for a in answers)
    total = len(answers)

    return render_template('test_result.html', participant=participant, correct_count=correct_count, total=total)

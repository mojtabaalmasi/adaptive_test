from flask import Flask, render_template, request, redirect, session, send_file
import sqlite3
import irt
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # حتما کلید امن انتخاب شود

DB_PATH = 'questions.db'  # مسیر فایل دیتابیس

# اتصال به دیتابیس
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# گرفتن تمام سوالات و پارامترهای IRT آنها
def get_all_questions():
    conn = get_db_connection()
    questions = conn.execute("SELECT id, text, a, b, c FROM questions").fetchall()
    conn.close()
    return questions

# گرفتن پارامترهای سوالات با ایندکس مشخص
def get_item_params_by_ids(ids, questions):
    params = []
    for q in questions:
        if q['id'] in ids:
            params.append((q['a'], q['b'], q['c']))
    return params

# ذخیره نتایج شرکت‌کننده در دیتابیس
def save_participant(name):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO participants (name) VALUES (?)", (name,))
    conn.commit()
    pid = cursor.lastrowid
    conn.close()
    return pid

def save_answer(participant_id, question_id, response):
    conn = get_db_connection()
    conn.execute("INSERT INTO answers (participant_id, question_id, response) VALUES (?, ?, ?)",
                 (participant_id, question_id, response))
    conn.commit()
    conn.close()

# صفحه ثبت‌نام و شروع آزمون
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        name = request.form.get('name')
        if not name:
            return render_template('index.html', error="لطفا نام خود را وارد کنید.")

        # ذخیره شرکت‌کننده و مقداردهی اولیه session
        participant_id = save_participant(name)
        session.clear()
        session['participant_id'] = participant_id
        session['name'] = name
        session['theta'] = 0.0
        session['responses'] = []
        session['answered_questions'] = []

        questions = get_all_questions()
        session['questions'] = [dict(q) for q in questions]

        next_q = select_next_question(session['theta'], session['questions'], session['answered_questions'])
        if next_q is None:
            return render_template('index.html', error="هیچ سوالی برای شروع آزمون یافت نشد.")
        
        session['current_question'] = next_q
        return redirect('/test')

    return render_template('index.html')

# تابع انتخاب سوال بعدی
def select_next_question(theta, questions, answered_ids):
    remaining = [q for q in questions if q['id'] not in answered_ids]
    if not remaining:
        return None

    best_q = None
    best_info = -1
    for q in remaining:
        info = irt.item_information(theta, q['a'], q['b'], q['c'])
        if info > best_info:
            best_info = info
            best_q = q
    return best_q

# صفحه آزمون
@app.route('/test', methods=['GET', 'POST'])
def test():
    if 'participant_id' not in session:
        return redirect('/')
    if 'current_question' not in session:
        return redirect('/')

    if request.method == 'POST':
        answer = request.form.get('option')
        if answer not in ['1', '2', '3', '4']:
            return render_template('test.html', question=session['current_question'], theta=session['theta'], error="لطفا یکی از گزینه‌ها را انتخاب کنید.")

        answer_int = int(answer)
        curr_q = session['current_question']

        save_answer(session['participant_id'], curr_q['id'], answer_int)

        session['responses'].append(answer_int)
        session['answered_questions'].append(curr_q['id'])

        answered_params = get_item_params_by_ids(session['answered_questions'], session['questions'])
        new_theta = irt.estimate_theta_mle(session['responses'], answered_params)
        session['theta'] = new_theta

        if len(session['answered_questions']) >= 15:
            return redirect('/result')

        next_q = select_next_question(session['theta'], session['questions'], session['answered_questions'])
        if next_q is None:
            return redirect('/result')

        session['current_question'] = next_q
        return redirect('/test')

    return render_template('test.html', question=session['current_question'], theta=session['theta'])

# صفحه نمایش نتیجه
@app.route('/result')
def result():
    if 'participant_id' not in session:
        return redirect('/')

    answered_params = get_item_params_by_ids(session['answered_questions'], session['questions'])
    word_path = f"results/result_{session['participant_id']}.docx"
    excel_path = f"results/result_{session['participant_id']}.xlsx"

    os.makedirs('results', exist_ok=True)

    irt.save_results_to_word(word_path, session['responses'], answered_params, session['theta'])
    irt.save_results_to_excel(excel_path, session['responses'], answered_params, session['theta'])

    return render_template('result.html', theta=session['theta'], word_file=word_path, excel_file=excel_path)

@app.route('/download/word')
def download_word():
    if 'participant_id' not in session:
        return redirect('/')
    word_path = f"results/result_{session['participant_id']}.docx"
    return send_file(word_path, as_attachment=True)

@app.route('/download/excel')
def download_excel():
    if 'participant_id' not in session:
        return redirect('/')
    excel_path = f"results/result_{session['participant_id']}.xlsx"
    return send_file(excel_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)

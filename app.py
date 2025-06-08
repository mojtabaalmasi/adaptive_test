from flask import Flask, render_template, request, redirect, session, send_file
import sqlite3
import irt
import os

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your_default_secret_key')

DB_PATH = 'questions.db'
MAX_QUESTIONS = 30  # تعداد سوالات آزمون

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_all_questions():
    conn = get_db_connection()
    questions = conn.execute("SELECT id, text, option1, option2, option3, option4, correct_option, a, b, c FROM questions").fetchall()
    conn.close()
    return questions

def get_item_params_by_ids(ids, questions):
    params = []
    for q in questions:
        if q['id'] in ids:
            params.append((q['a'], q['b'], q['c']))
    return params

def save_participant(data):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO participants (name, language, major, age, farsi_level, farsi_skills, farsi_courses , learning_place)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get('name'),
        
        data.get('language'),
        data.get('major'),
        int(data.get('age', 0)),
        data.get('farsi_level'),
        data.get('farsi_skills'),
		data.get('farsi_courses'),
        data.get('learning_place')
    ))
    conn.commit()
    pid = cursor.lastrowid
    conn.close()
    return pid

def save_answer(participant_id, question_id, selected_option):
    conn = get_db_connection()
    conn.execute("INSERT INTO answers (participant_id, question_id, selected_option ,is_correct) VALUES (?, ?, ?, ?)",
                 (participant_id, question_id, selected_option))
    conn.commit()
    conn.close()

@@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = {
            'name': request.form.get('name'),
            'language': request.form.get('language'),
            'major': request.form.get('major'),
            'age': request.form.get('age'),
            'farsi_level': request.form.get('farsi_level'),
            'farsi_skills': request.form.get('farsi_skills'),
            'farsi_courses': request.form.get('farsi_courses'),
            'learning_place': request.form.get('learning_place')
        }

        # اعتبارسنجی اولیه
        if not data['name']:
            return render_template('index.html', error="نام و نام خانوادگی الزامی است.", data=data)

        try:
            age = int(data['age'])
            if age <= 0 or age > 120:
                raise ValueError()
        except:
            return render_template('index.html', error="سن نامعتبر است.", data=data)

        participant_id = save_participant(data)
        session.clear()
        session['participant_id'] = participant_id
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


@app.route('/test', methods=['GET', 'POST'])
def test():
    if 'participant_id' not in session or 'current_question' not in session:
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

        if len(session['answered_questions']) >= MAX_QUESTIONS:
            return redirect('/result')

        next_q = select_next_question(session['theta'], session['questions'], session['answered_questions'])
        if next_q is None:
            return redirect('/result')

        session['current_question'] = next_q
        return redirect('/test')

    return render_template('test.html', question=session['current_question'], theta=session['theta'])

@app.route('/result')
def result():
    if 'participant_id' not in session:
        return redirect('/')

    answered_params = get_item_params_by_ids(session['answered_questions'], session['questions'])
    participant_id = session['participant_id']

    os.makedirs('results', exist_ok=True)

    word_path = f"results/result_{participant_id}.docx"
    excel_path = f"results/result_{participant_id}.xlsx"

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

# فایل app.py - نسخه کامل و بازنویسی‌شده
from flask import Flask, render_template, request, redirect, url_for, session
import os
import sqlite3
import math
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
from openpyxl.styles import Font
from docx import Document

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

DB_PATH = 'questions.db'
OUTPUT_FOLDER = os.path.join(os.getcwd(), 'static')

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# --- مدل 3PL ---
def irt_3pl_probability(theta, a, b, c):
    exp_part = math.exp(a * (theta - b))
    return c + (1 - c) * (exp_part / (1 + exp_part))

def item_information(theta, a, b, c):
    p = irt_3pl_probability(theta, a, b, c)
    q = 1 - p
    return (a ** 2) * ((q / p) * ((p - c) / (1 - c)) ** 2)

def estimate_theta_mle(responses, item_params):
    thetas = np.linspace(-4, 4, 81)
    likelihoods = []
    for theta in thetas:
        L = 1.0
        for u, (a, b, c) in zip(responses, item_params):
            p = irt_3pl_probability(theta, a, b, c)
            L *= p if u == 1 else (1 - p)
        likelihoods.append(L)
    return thetas[np.argmax(likelihoods)]

# --- بارگیری و انتخاب سوالات ---
def load_questions():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, text, a, b, c,
               option1, option2, option3, option4, correct_option
        FROM questions
    """)
    rows = cursor.fetchall()
    conn.close()
    return [{
        'id': row[0], 'text': row[1], 'a': float(row[2]), 'b': float(row[3]), 'c': float(row[4]),
        'options': [row[5], row[6], row[7], row[8]], 'correct': int(row[9])
    } for row in rows]

def select_next_question(theta, questions, asked_ids):
    candidates = [q for q in questions if q['id'] not in asked_ids]
    if not candidates:
        return None
    infos = [item_information(theta, q['a'], q['b'], q['c']) for q in candidates]
    return candidates[np.argmax(infos)]

# --- رسم نمودارها ---
def plot_icc(item_params, path):
    plt.figure()
    theta_range = np.linspace(-4, 4, 100)
    for i, (a, b, c) in enumerate(item_params):
        p = [irt_3pl_probability(t, a, b, c) for t in theta_range]
        plt.plot(theta_range, p, label=f'سوال {i+1}')
    plt.xlabel('θ')
    plt.ylabel('احتمال')
    plt.legend()
    plt.savefig(path)
    plt.close()

def plot_item_information(item_params, path):
    plt.figure()
    theta_range = np.linspace(-4, 4, 100)
    for i, (a, b, c) in enumerate(item_params):
        info = [item_information(t, a, b, c) for t in theta_range]
        plt.plot(theta_range, info, label=f'سوال {i+1}')
    plt.xlabel('θ')
    plt.ylabel('اطلاعات')
    plt.legend()
    plt.savefig(path)
    plt.close()

# --- ذخیره خروجی‌ها ---
def save_results_to_excel(path, responses, item_params, theta):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "نتایج"
    ws.append(['شماره', 'پاسخ', 'a', 'b', 'c'])
    for i, (r, (a, b, c)) in enumerate(zip(responses, item_params), 1):
        ws.append([i, r, a, b, c])
    ws.append([])
    ws.append(['θ تخمینی', theta])
    wb.save(path)

def save_results_to_word(path):
    doc = Document()
    doc.add_heading('مشخصات شرکت‌کننده', level=1)
    keys = ["full_name", "native_language", "major", "age", "persian_familiarity",
            "reading_level", "writing_level", "speaking_level", "listening_level",
            "persian_courses", "persian_institute"]
    for k in keys:
        doc.add_paragraph(f"{k}: {session.get(k, '')}")
    doc.save(path)

# --- مسیرها ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test', methods=['POST'])
def start_test():
    for k in request.form:
        session[k] = request.form[k]
    session['responses'] = []
    session['item_params'] = []
    session['asked_ids'] = []
    return redirect(url_for('show_question'))

@app.route('/question', methods=['GET', 'POST'])
def show_question():
    # در route مربوط به نمایش سوال (show_question)
    progress_percentage = int(100 * len(session['responses']) / MAX_QUESTIONS)
    return render_template("test.html",
                           question=next_q,
                           question_number=len(session['responses']) + 1,
                           progress_percentage=progress_percentage)

    if request.method == 'POST':
        selected = int(request.form.get('answer'))
        correct = session['current_question']['correct']
        session['responses'].append(int(selected == correct))
        session['item_params'].append([
            session['current_question']['a'],
            session['current_question']['b'],
            session['current_question']['c']
        ])
        session['asked_ids'].append(session['current_question']['id'])

    questions = load_questions()
    theta = estimate_theta_mle(session['responses'], session['item_params']) if session['responses'] else 0
    next_q = select_next_question(theta, questions, session['asked_ids'])
    if not next_q:
        session['theta'] = theta
        return redirect(url_for('result'))
    session['current_question'] = next_q
    return render_template('question.html', question=next_q)

@app.route('/results')
def result():
    theta = session.get('theta', 0)
    responses = session.get('responses', [])
    item_params = session.get('item_params', [])

    name = session['full_name'].replace(' ', '_')
    excel_file = f'result_{name}.xlsx'
    word_file = f'result_{name}.docx'
    icc_file = f'icc_{name}.png'
    info_file = f'info_{name}.png'

    save_results_to_excel(os.path.join(OUTPUT_FOLDER, excel_file), responses, item_params, theta)
    save_results_to_word(os.path.join(OUTPUT_FOLDER, word_file))
    plot_icc(item_params, os.path.join(OUTPUT_FOLDER, icc_file))
    plot_item_information(item_params, os.path.join(OUTPUT_FOLDER, info_file))

    return render_template('result.html',
                           full_name=session.get('full_name'),
                           native_language=session.get('native_language'),
                           major=session.get('major'),
                           age=session.get('age'),
                           persian_familiarity=session.get('persian_familiarity'),
                           reading_level=session.get('reading_level', ''),
                           writing_level=session.get('writing_level', ''),
                           speaking_level=session.get('speaking_level', ''),
                           listening_level=session.get('listening_level', ''),
                           persian_courses=session.get('persian_courses', ''),
                           persian_institute=session.get('persian_institute', ''),
                           theta=theta,
                           excel_file=url_for('static', filename=excel_file),
                           word_file=url_for('static', filename=word_file),
                           icc_image=url_for('static', filename=icc_file),
                           info_image=url_for('static', filename=info_file))

if __name__ == '__main__':
    app.run(debug=True)
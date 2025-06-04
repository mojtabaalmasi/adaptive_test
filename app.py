from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
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

# --- توابع مدل 3PL ---

def irt_3pl_probability(theta, a, b, c):
    exp_part = math.exp(a * (theta - b))
    p = c + (1 - c) * (exp_part / (1 + exp_part))
    return p

def item_information(theta, a, b, c):
    p = irt_3pl_probability(theta, a, b, c)
    q = 1 - p
    info = (a ** 2) * ((q / p) * ((p - c) / (1 - c)) ** 2)
    return info

def estimate_theta_mle(responses, item_params):
    thetas = np.linspace(-4, 4, 81)
    likelihoods = []

    for theta in thetas:
        L = 1.0
        for u, (a, b, c) in zip(responses, item_params):
            p = irt_3pl_probability(theta, a, b, c)
            L *= p if u == 1 else (1 - p)
        likelihoods.append(L)

    max_idx = np.argmax(likelihoods)
    return thetas[max_idx]

# --- توابع مربوط به سوالات و انتخاب سوال بعدی ---

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

    questions = [{
        'id': row[0],
        'text': row[1],
        'a': float(row[2]),
        'b': float(row[3]),
        'c': float(row[4]),
        'options': [row[5], row[6], row[7], row[8]],
        'correct': int(row[9])
    } for row in rows]
    return questions

def select_next_question(theta, questions, asked_ids):
    candidates = [q for q in questions if q['id'] not in asked_ids]
    if not candidates:
        return None
    infos = [item_information(theta, q['a'], q['b'], q['c']) for q in candidates]
    max_idx = np.argmax(infos)
    return candidates[max_idx]

# --- توابع تولید نمودارها ---

def plot_icc(item_params, save_path):
    plt.figure(figsize=(8,6))
    theta_range = np.linspace(-4, 4, 100)
    for i, (a, b, c) in enumerate(item_params):
        p = [irt_3pl_probability(t, a, b, c) for t in theta_range]
        plt.plot(theta_range, p, label=f'سوال {i+1}')
    plt.xlabel('توانایی θ')
    plt.ylabel('احتمال پاسخ صحیح')
    plt.title('نمودار تابع احتمال شرطی (ICC)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_item_information(item_params, save_path):
    plt.figure(figsize=(8,6))
    theta_range = np.linspace(-4, 4, 100)
    for i, (a, b, c) in enumerate(item_params):
        info = [item_information(t, a, b, c) for t in theta_range]
        plt.plot(theta_range, info, label=f'سوال {i+1}')
    plt.xlabel('توانایی θ')
    plt.ylabel('اطلاعات آیتم')
    plt.title('نمودار اطلاعات آیتم')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# --- ذخیره نتایج ---

def save_results_to_excel(file_path, responses, item_params, theta):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "نتایج آزمون"

    headers = ['شماره سوال', 'پاسخ داده شده', 'a', 'b', 'c']
    ws.append(headers)
    for cell in ws[1]:
        cell.font = Font(bold=True)

    for i, (resp, (a, b, c)) in enumerate(zip(responses, item_params), start=1):
        ws.append([i, resp, a, b, c])

    ws.append([])
    ws.append(["توانایی تخمینی θ:", theta])

    info_ws = wb.create_sheet("مشخصات فردی")
    info_ws.append(["کلید", "مقدار"])
    for cell in info_ws[1]:
        cell.font = Font(bold=True)

    personal_info_keys = [
        ("نام و نام خانوادگی", 'full_name'),
        ("زبان مادری", 'native_language'),
        ("رشته تحصیلی", 'major'),
        ("سن", 'age'),
        ("آشنایی با زبان فارسی", 'persian_familiarity'),
        ("سطح خواندن", 'reading_level'),
        ("سطح نوشتن", 'writing_level'),
        ("سطح صحبت کردن", 'speaking_level'),
        ("سطح شنیداری", 'listening_level'),
        ("دوره‌های زبان فارسی", 'persian_courses'),
        ("محل آموزش زبان فارسی", 'persian_institute'),
    ]
    for label, key in personal_info_keys:
        info_ws.append([label, session.get(key, '')])

    wb.save(file_path)

# --- صفحه ثبت‌نام ---

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/test', methods=['POST'])
def start_test():
    data = request.form
    session['full_name'] = data.get('full_name')
    session['native_language'] = data.get('native_language')
    session['major'] = data.get('major')
    session['age'] = data.get('age')
    session['persian_familiarity'] = data.get('persian_familiarity')

    session['responses'] = []
    session['item_params'] = []
    session['asked_ids'] = []

    return redirect(url_for('show_question'))

@app.route('/test', methods=['GET', 'POST'])
def show_question():
    if request.method == 'POST':
        selected = int(request.form.get('answer'))
        correct = session['current_question']['correct']
        response = int(selected == correct)

        session['responses'].append(response)
        session['item_params'].append([
            session['current_question']['a'],
            session['current_question']['b'],
            session['current_question']['c']
        ])
        session['asked_ids'].append(session['current_question']['id'])

    questions = load_questions()
    if not session['responses']:
        theta = 0
    else:
        theta = estimate_theta_mle(session['responses'], session['item_params'])

    next_q = select_next_question(theta, questions, session['asked_ids'])
    if next_q is None:
        session['theta'] = theta
        return redirect(url_for('result'))

    session['current_question'] = next_q
    return render_template('question.html', question=next_q)

@app.route('/results')
def result():
    theta = session.get('theta', 0)
    responses = session.get('responses', [])
    item_params = session.get('item_params', [])

    filename = f"result_{session['full_name'].replace(' ', '_')}.xlsx"
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    save_results_to_excel(file_path, responses, item_params, theta)

    return render_template('result.html', theta=theta, file_link=url_for('static', filename=filename))

if __name__ == '__main__':
    app.run(debug=True)

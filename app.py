from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
import os
import sqlite3
import math
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

DB_PATH = 'questions.db'
OUTPUT_FOLDER = os.path.join(os.getcwd(), 'static')
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# توابع مدل 3PL
def irt_3pl_probability(theta, a, b, c):
    exp_part = math.exp(a * (theta - b))
    p = c + (1 - c) * (exp_part / (1 + exp_part))
    return p

def item_information(theta, a, b, c):
    p = irt_3pl_probability(theta, a, b, c)
    q = 1 - p
    info = (a ** 2) * ((q / p) * ((p - c) / (1 - c)) ** 2)
    return info

# تخمین توانایی ساده با MLE عددی (جستجوی گرید)
def estimate_theta_mle(responses, item_params):
    # پاسخ ها لیستی از 0 و 1 است، item_params لیستی از (a,b,c)
    # جستجوی گرید بین -4 تا 4
    thetas = np.linspace(-4, 4, 81)  # گام 0.1
    likelihoods = []

    for theta in thetas:
        L = 1.0
        for u, (a, b, c) in zip(responses, item_params):
            p = irt_3pl_probability(theta, a, b, c)
            L *= p if u == 1 else (1 - p)
        likelihoods.append(L)

    max_idx = np.argmax(likelihoods)
    return thetas[max_idx]

def load_questions():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, text, a, b, c FROM questions")
    rows = cursor.fetchall()
    conn.close()
    questions = [{'id': row[0], 'text': row[1], 'a': row[2], 'b': row[3], 'c': row[4]} for row in rows]
    return questions

def select_next_question(theta, questions, asked_ids):
    # سوالی را که بیشترین اطلاعات در theta دارد ولی پرسیده نشده انتخاب کن
    candidates = [q for q in questions if q['id'] not in asked_ids]
    if not candidates:
        return None
    infos = [item_information(theta, q['a'], q['b'], q['c']) for q in candidates]
    max_idx = np.argmax(infos)
    return candidates[max_idx]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test', methods=['GET', 'POST'])
def test():
    questions = load_questions()

    if request.method == 'POST' and 'name' in request.form:
        # شروع آزمون با گرفتن اطلاعات اولیه
        session['name'] = request.form['name']
        session['phone'] = request.form['phone']
        session['major'] = request.form['major']
        session['responses'] = []
        session['asked_questions'] = []
        session['theta'] = 0.0  # توانایی شروعی
        # انتخاب سوال اول با بیشترین اطلاعات در theta=0
        first_q = select_next_question(0.0, questions, [])
        if first_q is None:
            return "هیچ سوالی برای آزمون وجود ندارد."
        session['current_question'] = first_q
        return render_template('test.html', questions=[first_q], step_number=1, total_steps='نامشخص')

    elif request.method == 'POST':
        # دریافت پاسخ سوال فعلی
        current_q = session.get('current_question', None)
        if current_q is None:
            return redirect(url_for('index'))

        ans = request.form.get(f'q{current_q["id"]}')
        if ans is None:
            return "لطفا یک پاسخ انتخاب کنید."

        responses = session.get('responses', [])
        asked = session.get('asked_questions', [])
        theta = session.get('theta', 0.0)

        responses.append(int(ans))
        asked.append(current_q['id'])

        # بروزرسانی توانایی
        item_params = [(q['a'], q['b'], q['c']) for q in questions if q['id'] in asked]
        theta = estimate_theta_mle(responses, item_params)
        session['theta'] = theta
        session['responses'] = responses
        session['asked_questions'] = asked

        # انتخاب سوال بعدی
        next_q = select_next_question(theta, questions, asked)
        if next_q is None:
            # آزمون تمام شده
            session['current_question'] = None
            # می‌تونی اینجا تحلیل‌ها و ذخیره‌سازی نتایج رو انجام بدی
            return redirect(url_for('results'))

        session['current_question'] = next_q
        step_number = len(asked) + 1
        return render_template('test.html', questions=[next_q], step_number=step_number, total_steps='نامشخص')

    return redirect(url_for('index'))

@app.route('/results')
def results():
    theta = session.get('theta', None)
    if theta is None:
        return redirect(url_for('index'))
    return render_template('results.html', theta=theta)

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for, session
import os
import sqlite3
from irt import (
    estimate_theta_mle,
    item_information,
    plot_icc,
    plot_item_information,
    save_results_to_excel,
    save_results_to_word
)

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

DB_PATH = 'questions.db'
OUTPUT_FOLDER = os.path.join(os.getcwd(), 'static')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- بارگذاری سؤالات از دیتابیس ---
def load_questions():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, text, a, b, c FROM questions")
    rows = cursor.fetchall()
    conn.close()
    return [{'id': r[0], 'text': r[1], 'a': r[2], 'b': r[3], 'c': r[4]} for r in rows]

# --- انتخاب سؤال با بیشترین اطلاعات ---
def select_next_question(questions, answered_ids, theta):
    remaining = [q for q in questions if q['id'] not in answered_ids]
    if not remaining:
        return None
    info_values = [item_information(theta, q['a'], q['b'], q['c']) for q in remaining]
    max_index = info_values.index(max(info_values))
    return remaining[max_index]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        session['name'] = request.form['name']
        session['phone'] = request.form['phone']
        session['major'] = request.form['major']
        session['responses'] = []
        session['answered_ids'] = []
        session['theta'] = 0.0
        return redirect(url_for('test'))
    return render_template('index.html')

@app.route('/test', methods=['GET', 'POST'])
def test():
    questions = load_questions()
    responses = session.get('responses', [])
    answered_ids = session.get('answered_ids', [])
    theta = session.get('theta', 0.0)

    if request.method == 'POST':
        qid = int(request.form['qid'])
        ans = int(request.form['answer'])
        responses.append((qid, ans))
        answered_ids.append(qid)

        item_params = [(q['a'], q['b'], q['c']) for q in questions if q['id'] in answered_ids]
        response_vals = [r[1] for r in responses]
        theta = estimate_theta_mle(response_vals, item_params)

        session['responses'] = responses
        session['answered_ids'] = answered_ids
        session['theta'] = theta

    next_q = select_next_question(questions, answered_ids, theta)
    if not next_q or len(answered_ids) >= 20:
        return redirect(url_for('results'))

    return render_template('test.html', question=next_q, step_number=len(answered_ids)+1, total_steps='Adaptive')

@app.route('/results')
def results():
    theta = session.get('theta')
    responses = session.get('responses', [])
    answered_ids = [r[0] for r in responses]
    questions = load_questions()
    used_items = [q for q in questions if q['id'] in answered_ids]
    item_params = [(q['a'], q['b'], q['c']) for q in used_items]
    response_vals = [r[1] for r in responses]

    plot_icc(item_params, save_path=os.path.join(OUTPUT_FOLDER, 'icc.png'))
    plot_item_information(item_params, save_path=os.path.join(OUTPUT_FOLDER, 'item_info.png'))
    save_results_to_excel(os.path.join(OUTPUT_FOLDER, 'results.xlsx'), response_vals, item_params, theta)
    save_results_to_word(os.path.join(OUTPUT_FOLDER, 'results.docx'), response_vals, item_params, theta)

    return render_template(
        'results.html',
        theta=theta,
        icc_image=url_for('static', filename='icc.png'),
        info_image=url_for('static', filename='item_info.png'),
        excel_file=url_for('static', filename='results.xlsx'),
        word_file=url_for('static', filename='results.docx'),
    )

if __name__ == '__main__':
    app.run(debug=True)

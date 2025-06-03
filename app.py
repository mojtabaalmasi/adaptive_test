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

    # مرحله اول: ورود اطلاعات اولیه
    if request.method == 'POST' and 'name' in request.form:
        session['name'] = request.form['name']
        session['phone'] = request.form['phone']
        session['major'] = request.form['major']
        session['responses'] = []
        session['asked_questions'] = []
        session['question_index'] = 0
        return render_template('test.html', questions=[questions[0]], step_number=1, total_steps='?')

    # مرحله دوم: پاسخ به سؤال
    elif request.method == 'POST' and 'q0' in request.form:
        responses = session.get('responses', [])
        asked = session.get('asked_questions', [])
        index = session.get('question_index', 0)

        # ذخیره پاسخ فعلی
        q_id = questions[index]['id']
        ans = request.form.get(f'q{q_id}')
        responses.append(int(ans) if ans == '1' else 0)
        asked.append(q_id)

        # برو به سؤال بعدی اگر باقی‌مانده
        index += 1
        if index < len(questions):
            session['responses'] = responses
            session['asked_questions'] = asked
            session['question_index'] = index
            return render_template('test.html', questions=[questions[index]], step_number=index+1, total_steps=len(questions))
        else:
            # آزمون تمام شده
            item_params = [(q['a'], q['b'], q['c']) for q in questions]
            theta = estimate_theta_mle(responses, item_params)
            session['theta'] = theta

            plot_icc(item_params, save_path=os.path.join(OUTPUT_FOLDER, 'icc.png'))
            plot_item_information(item_params, save_path=os.path.join(OUTPUT_FOLDER, 'item_info.png'))
            save_results_to_excel(os.path.join(OUTPUT_FOLDER, 'results.xlsx'), responses, item_params, theta)
            save_results_to_word(os.path.join(OUTPUT_FOLDER, 'results.docx'), responses, item_params, theta)

            return redirect(url_for('results'))

    return redirect(url_for('index'))



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

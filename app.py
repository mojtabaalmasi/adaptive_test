from flask import Flask, render_template, request, redirect, session, send_file, url_for
import sqlite3
import numpy as np
import pandas as pd
from docx import Document
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # کلید امن را حتما تغییر دهید

DATABASE = 'your_db_name.db'  # نام دیتابیس شما

# توابع مربوط به دیتابیس
def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def get_question_by_id(question_id):
    conn = get_db_connection()
    question = conn.execute(
        "SELECT id, question_text, option1, option2, option3, option4 FROM questions WHERE id = ?", 
        (question_id,)
    ).fetchone()
    conn.close()
    if question:
        return {
            'id': question['id'],
            'text': question['question_text'],
            'options': [question['option1'], question['option2'], question['option3'], question['option4']]
        }
    return None

def get_all_item_params():
    conn = get_db_connection()
    rows = conn.execute("SELECT a, b, c FROM questions ORDER BY id").fetchall()
    conn.close()
    return [(row['a'], row['b'], row['c']) for row in rows]

# توابع مدل 3PL
def three_pl_probability(theta, a, b, c):
    return c + (1 - c) / (1 + np.exp(-a * (theta - b)))

def estimate_theta_mle(responses, item_params, lr=0.01, max_iter=500, tol=1e-5):
    theta = 0.0
    for _ in range(max_iter):
        grad = 0
        for i, (a, b, c) in enumerate(item_params):
            p = three_pl_probability(theta, a, b, c)
            q = 1 - p
            dL = a * (responses[i] - p) * (1 - c) / (p * q + 1e-9)
            grad += dL
        theta_new = theta + lr * grad
        if abs(theta_new - theta) < tol:
            break
        theta = theta_new
    return theta

def item_information(theta, a, b, c):
    p = three_pl_probability(theta, a, b, c)
    q = 1 - p
    return (a ** 2) * ((p - c) ** 2) / ((1 - c) ** 2 * p * q + 1e-9)

def select_next_question(theta, all_item_params, answered_indices):
    infos = []
    for i, (a, b, c) in enumerate(all_item_params):
        if i in answered_indices:
            infos.append(-np.inf)
        else:
            infos.append(item_information(theta, a, b, c))
    next_q = np.argmax(infos)
    if infos[next_q] == -np.inf:
        return None
    return next_q

def plot_icc(item_params, save_path='static/icc.png'):
    theta_range = np.linspace(-4, 4, 100)
    plt.figure(figsize=(10, 6))
    for i, (a, b, c) in enumerate(item_params):
        probs = [three_pl_probability(t, a, b, c) for t in theta_range]
        plt.plot(theta_range, probs, label=f"سوال {i+1}")
    plt.xlabel('θ (توانایی)')
    plt.ylabel('احتمال پاسخ صحیح')
    plt.title('نمودار تابع مشخصه سوالات (ICC)')
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path

def plot_item_information(item_params, save_path='static/item_info.png'):
    theta_range = np.linspace(-4, 4, 100)
    total_info = np.zeros_like(theta_range)
    for a, b, c in item_params:
        info = [item_information(t, a, b, c) for t in theta_range]
        total_info += info
    plt.figure(figsize=(8, 5))
    plt.plot(theta_range, total_info, color='darkblue')
    plt.xlabel('θ (توانایی)')
    plt.ylabel('اطلاعات آزمون')
    plt.title('نمودار تابع اطلاعات آزمون')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path

def save_results_to_excel(filepath, responses, answered_indices, theta):
    conn = get_db_connection()
    rows = conn.execute("SELECT id, question_text, option1, option2, option3, option4, a, b, c FROM questions ORDER BY id").fetchall()
    conn.close()

    data = []
    for i, resp in zip(answered_indices, responses):
        row = rows[i]
        data.append({
            'سوال': row['question_text'],
            'پاسخ': resp,
            'a': row['a'],
            'b': row['b'],
            'c': row['c']
        })

    df = pd.DataFrame(data)
    df.loc[len(df.index)] = ['θ (توانایی)', theta, '', '', '']
    df.to_excel(filepath, index=False)

def save_results_to_word(filepath, responses, answered_indices, theta):
    conn = get_db_connection()
    rows = conn.execute("SELECT id, question_text, option1, option2, option3, option4, a, b, c FROM questions ORDER BY id").fetchall()
    conn.close()

    doc = Document()
    doc.add_heading('نتایج آزمون انطباقی (3PL)', level=1)
    doc.add_paragraph(f'مقدار تخمینی θ: {theta:.3f}')
    table = doc.add_table(rows=1, cols=5)
    hdr = table.rows[0].cells
    hdr[0].text = 'سوال'
    hdr[1].text = 'پاسخ'
    hdr[2].text = 'a'
    hdr[3].text = 'b'
    hdr[4].text = 'c'

    for i, resp in zip(answered_indices, responses):
        row = rows[i]
        cells = table.add_row().cells
        cells[0].text = row['question_text']
        cells[1].text = str(resp)
        cells[2].text = f"{row['a']:.2f}"
        cells[3].text = f"{row['b']:.2f}"
        cells[4].text = f"{row['c']:.2f}"

    doc.save(filepath)

@app.route('/')
def index():
    session.clear()
    session['answered_questions'] = []
    session['responses'] = []
    session['theta'] = 0.0
    return render_template('index.html')

@app.route('/test', methods=['GET', 'POST'])
def test():
    if 'answered_questions' not in session:
        return redirect(url_for('index'))

    answered = session['answered_questions']
    responses = session['responses']
    theta = session.get('theta', 0.0)

    all_item_params = get_all_item_params()
    total_questions = len(all_item_params)

    if request.method == 'POST':
        selected_option = request.form.get('option')
        if selected_option is None:
            # پیام خطا برای انتخاب نکردن گزینه
            last_q_id = answered[-1]
            question = get_question_by_id(last_q_id + 1)
            return render_template('test.html', question=question, error="لطفا یک گزینه را انتخاب کنید.")

        answer_int = int(selected_option)
        responses.append(answer_int)
        session['responses'] = responses

        # تخمین θ با پاسخ‌های فعلی
        answered_params = [all_item_params[i] for i in answered]
        theta = estimate_theta_mle(responses, answered_params)
        session['theta'] = theta

    # انتخاب سوال بعدی
    next_q = select_next_question(theta, all_item_params, answered)
    if next_q is None or len(answered) >= total_questions:
        return redirect(url_for('result'))

    # افزودن سوال جدید به لیست پاسخ داده شده (اگر هنوز اضافه نشده)
    if len(answered) < len(responses):
        # یعنی الان سوال قبلی جواب داده شده، پس سوال جدید را اضافه می‌کنیم
        answered.append(next_q)
    elif len(answered) == 0:
        answered.append(next_q)

    session['answered_questions'] = answered
    session.modified = True

    question = get_question_by_id(next_q + 1)  # چون id سوالات در دیتابیس معمولاً از 1 شروع می‌شود

    return render_template('test.html', question=question)

@app.route('/result')
def result():
    if 'responses' not in session or 'answered_questions' not in session:
        return redirect(url_for('index'))

    responses = session['responses']
    answered = session['answered_questions']
    theta = session.get('theta', 0.0)

    all_item_params = get_all_item_params()
    answered_params = [all_item_params[i] for i in answered]

    icc_path = plot_icc(answered_params, save_path='static/icc.png')
    info_path = plot_item_information(answered_params, save_path='static/item_info.png')

    return render_template('result.html', theta=theta, icc_image=icc_path, info_image=info_path)

@app.route('/download/<filetype>')
def download(filetype):
    if 'responses' not in session or 'answered_questions' not in session:
        return redirect(url_for('index'))

    responses = session['responses']
    answered = session['answered_questions']
    theta = session.get('theta', 0.0)

    if filetype == 'excel':
        filepath = 'results.xlsx'
        save_results_to_excel(filepath, responses, answered, theta)
        return send_file(filepath, as_attachment=True)
    elif filetype == 'word':
        filepath = 'results.docx'
        save_results_to_word(filepath, responses, answered, theta)
        return send_file(filepath, as_attachment=True)
    else:
        return redirect(url_for('result'))

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)

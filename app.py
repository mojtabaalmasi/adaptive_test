from flask import Flask, render_template, request, redirect, session, send_file, url_for
import sqlite3
import numpy as np
import pandas as pd
from docx import Document
import matplotlib.pyplot as plt
import os
import uuid


app = Flask(__name__)
app.secret_key = os.urandom(24)  # کلید مخفی ایمن
app.secret_key = 'یک_کلید_محرمانه_و_دلخواه_اینجا_قرار_ده'


DATABASE = 'questions.db'

# ------------------ توابع دیتابیس ------------------

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn



def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db

# همچنین برای بستن اتصال در پایان درخواست:


def get_question_by_id(question_id):
    conn = get_db_connection()
    question = conn.execute("SELECT id, text, option1, option2, option3, option4 FROM questions WHERE id = ?", (question_id,)).fetchone()
    conn.close()
    if question:
        return {
            'id': int(question['id']),
            'text': question['text'],
            'options': [question['option1'], question['option2'], question['option3'], question['option4']]
        }
    return None

def get_all_item_params():
    conn = get_db_connection()
    rows = conn.execute("SELECT a, b, c FROM questions ORDER BY id").fetchall()
    conn.close()
    return [(float(row['a']), float(row['b']), float(row['c'])) for row in rows]

# ------------------ توابع IRT ------------------

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
    return float(theta)

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
    next_q = int(np.argmax(infos))
    return None if infos[next_q] == -np.inf else next_q

# ------------------ نمودار ------------------

def plot_icc(item_params, save_path):
    theta_range = np.linspace(-4, 4, 100)
    plt.figure(figsize=(10, 6))
    for i, (a, b, c) in enumerate(item_params):
        probs = [three_pl_probability(t, a, b, c) for t in theta_range]
        plt.plot(theta_range, probs, label=f"سوال {i+1}")
    plt.xlabel('θ (توانایی)')
    plt.ylabel('احتمال پاسخ صحیح')
    plt.title('تابع مشخصه سوالات (ICC)')
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path

def plot_item_information(item_params, save_path):
    theta_range = np.linspace(-4, 4, 100)
    total_info = np.zeros_like(theta_range)
    for a, b, c in item_params:
        info = [item_information(t, a, b, c) for t in theta_range]
        total_info += info
    plt.figure(figsize=(8, 5))
    plt.plot(theta_range, total_info, color='darkblue')
    plt.xlabel('θ (توانایی)')
    plt.ylabel('اطلاعات آزمون')
    plt.title('تابع اطلاعات کل آزمون')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path

# ------------------ ذخیره نتایج ------------------

def save_results_to_excel(filepath, responses, answered_indices, theta):
    conn = get_db_connection()
    rows = conn.execute("SELECT text, a, b, c FROM questions ORDER BY id").fetchall()
    conn.close()
    data = []
    for i, r in zip(answered_indices, responses):
        row = rows[i]
        data.append({
            'سوال': row['text'],
            'پاسخ': r,
            'a': row['a'],
            'b': row['b'],
            'c': row['c']
        })
    df = pd.DataFrame(data)
    df.loc[len(df)] = ['θ (توانایی)', theta, '', '', '']
    df.to_excel(filepath, index=False)

def save_results_to_word(filepath, responses, answered_indices, theta):
    conn = get_db_connection()
    rows = conn.execute("SELECT text, a, b, c FROM questions ORDER BY id").fetchall()
    conn.close()
    doc = Document()
    doc.add_heading('نتایج آزمون تطبیقی', 0)
    doc.add_paragraph(f'مقدار تخمینی θ: {theta:.3f}')
    table = doc.add_table(rows=1, cols=5)
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'سوال'
    hdr_cells[1].text = 'پاسخ'
    hdr_cells[2].text = 'a'
    hdr_cells[3].text = 'b'
    hdr_cells[4].text = 'c'
    for i, r in zip(answered_indices, responses):
        row = rows[i]
        cells = table.add_row().cells
        cells[0].text = row['text']
        cells[1].text = str(r)
        cells[2].text = str(row['a'])
        cells[3].text = str(row['b'])
        cells[4].text = str(row['c'])
    doc.save(filepath)

# ------------------ مسیرها ------------------

@app.route('/')
def index():
    session.clear()
    session['answered_questions'] = []
    session['responses'] = []
    session['theta'] = 0.0
    return render_template('index.html')

@app.route('/test', methods=['GET', 'POST'])
def test():
    # --- 1. بررسی اینکه کاربر ثبت‌نام کرده و participant_id در session هست ---
    if 'participant_id' not in session:
        return redirect(url_for('index'))

    # --- 2. بررسی وجود answered_questions در session ---
    if 'answered_questions' not in session:
        return redirect(url_for('index'))

    # --- 3. دریافت سوالات جواب داده شده و پاسخ‌ها از session ---
    answered = list(map(int, session.get('answered_questions', [])))
    responses = list(map(int, session.get('responses', [])))

    # --- 4. دریافت مقدار θ (توانایی آزمون‌دهنده) ---
    theta = float(session.get('theta', 0.0))

    # --- 5. دریافت پارامترهای همه سوالات ---
    all_item_params = get_all_item_params()
    total_questions = len(all_item_params)

    if request.method == 'POST':
        selected_option = request.form.get('answer')
        # --- 6. اگر پاسخ انتخاب نشده، ارور نمایش بده ---
        if selected_option is None:
            current_q_index = answered[-1] if answered else 0
            question = get_question_by_id(current_q_index)
            progress = int(len(answered) / total_questions * 100)
            return render_template('test.html', question=question, error="لطفا یک گزینه را انتخاب کنید.", progress=progress)

        # --- 7. ذخیره پاسخ جدید ---
        responses.append(int(selected_option))
        answered_params = [all_item_params[i] for i in answered]
        theta = estimate_theta_mle(responses, answered_params)

        # --- ذخیره پاسخ در دیتابیس ---
        db = get_db_connection()
        cursor = db.cursor()
        participant_id = session['participant_id']
        current_question_id = answered[-1] + 1  # اگر id سوال‌ها 1-indexed است

        cursor.execute(
            "INSERT INTO answers (user_id, question_id, response) VALUES (?, ?, ?)",
            (participant_id, current_question_id, int(selected_option))
        )
        db.commit()

        # --- 8. انتخاب سوال بعدی ---
        next_q = select_next_question(theta, all_item_params, answered)
        
        # --- 9. اگر سوالی باقی نمانده یا همه سوالات جواب داده شده ---
        if next_q is None or len(answered) >= total_questions:
            # ذخیره نهایی θ در session
            session['theta'] = float(theta)
            session['answered_questions'] = list(map(int, answered))
            session['responses'] = list(map(int, responses))

            # --- ذخیره نمره نهایی در جدول user_results ---
            # ابتدا چک کن آیا رکوردی از قبل هست یا نه (برای به‌روزرسانی)
            cursor.execute("SELECT id FROM user_results WHERE user_id = ?", (participant_id,))
            existing = cursor.fetchone()
            if existing:
                cursor.execute("UPDATE user_results SET theta = ? WHERE user_id = ?", (theta, participant_id))
            else:
                cursor.execute("INSERT INTO user_results (user_id, theta) VALUES (?, ?)", (participant_id, theta))
            db.commit()

            return redirect(url_for('result'))

        # --- 10. آماده‌سازی برای سوال بعد ---
        answered.append(next_q)
        session['answered_questions'] = list(map(int, answered))
        session['responses'] = list(map(int, responses))
        session['theta'] = float(theta)
        session.modified = True

        question = get_question_by_id(next_q + 1)
        progress = int(len(answered) / total_questions * 100)
        return render_template('test.html', question=question, progress=progress)

    # --- 11. حالت GET: شروع آزمون یا ادامه آزمون ---
    if not answered:
        next_q = 0
        answered.append(next_q)
        session['answered_questions'] = list(map(int, answered))
        session.modified = True
    else:
        next_q = answered[-1]

    question = get_question_by_id(next_q + 1)
    progress = int(len(answered) / total_questions * 100)
    return render_template('test.html', question=question, progress=progress)





@app.route('/result')
def result():
    if 'responses' not in session or 'answered_questions' not in session:
        return redirect(url_for('index'))

    responses = list(map(int, session['responses']))
    answered = list(map(int, session['answered_questions']))
    theta = float(session.get('theta', 0.0))
    all_item_params = get_all_item_params()
    answered_params = [all_item_params[i] for i in answered]

    icc_path = plot_icc(answered_params, save_path=f'static/icc_{uuid.uuid4().hex}.png')
    info_path = plot_item_information(answered_params, save_path=f'static/info_{uuid.uuid4().hex}.png')

    user_name = session.get('name', 'کاربر ناشناس')

    # تحلیل ساده بر اساس θ
    if theta < -1:
        interpretation = "توانایی شما پایین‌تر از حد متوسط است."
    elif -1 <= theta <= 1:
        interpretation = "توانایی شما در سطح متوسط قرار دارد."
    else:
        interpretation = "توانایی شما بالاتر از حد متوسط است."

    return render_template('result.html',
                           theta=theta,
                           user_name=user_name,
                           icc_image=icc_path,
                           info_image=info_path,
                           interpretation=interpretation)

@app.route('/register', methods=['POST'])
def register():
          
    name = request.form.get('name')
    session['user_name'] = name
    age = request.form.get('age')
    language = request.form.get('language')
    major = request.form.get('major')
    farsi_level = request.form.get('farsi_level')
    farsi_skills = request.form.get('farsi_skills')
    farsi_courses = request.form.get('farsi_courses')
    learning_place = request.form.get('learning_place')

    conn = sqlite3.connect('questions.db')
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO participants (name, age, language, major, farsi_level, farsi_skills, farsi_courses, learning_place)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (name, age, language, major, farsi_level, farsi_skills, farsi_courses, learning_place))
    conn.commit()

    # گرفتن آیدی آخرین ثبت شده
    participant_id = cursor.lastrowid
    conn.close()

    # ذخیره آیدی در session برای استفاده در صفحات بعدی
    session['participant_id'] = participant_id
    
    # فرض کنیم نام کاربر رو از فرم ثبت‌نام می‌گیریم
    name = request.form['name']
    session['user_name'] = name


    # هدایت به صفحه آزمون (مثلاً /test)
    return redirect(url_for('test'))




@app.route('/download/<filetype>')
def download(filetype):
    if 'responses' not in session or 'answered_questions' not in session:
        return redirect(url_for('index'))

    responses = list(map(int, session['responses']))
    answered = list(map(int, session['answered_questions']))
    theta = float(session.get('theta', 0.0))

    filename = f'results_{uuid.uuid4().hex}'
    if filetype == 'excel':
        filepath = f'static/{filename}.xlsx'
        save_results_to_excel(filepath, responses, answered, theta)
        return send_file(filepath, as_attachment=True)
    elif filetype == 'word':
        filepath = f'static/{filename}.docx'
        save_results_to_word(filepath, responses, answered, theta)
        return send_file(filepath, as_attachment=True)
    else:
        return redirect(url_for('result'))

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)

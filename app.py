from flask import Flask,flash, render_template, request, redirect, session, send_file, url_for
import sqlite3
import numpy as np
import pandas as pd
from docx import Document
import matplotlib.pyplot as plt
import os
import uuid


app = Flask(__name__)
app.secret_key = 'your_secret_key'

DATABASE = 'questions.db'
def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS participants (
                participant_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                nationality TEXT,
                native_language TEXT,
                official_language TEXT,
                age INTEGER,
                field_of_study TEXT,
                education_level TEXT,
                job TEXT,
                role TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS teacher_info (
                participant_id INTEGER,
                teaching_years INTEGER,
                teaching_institutions TEXT,
                teaching_level TEXT,
                teacher_comment TEXT,
                FOREIGN KEY(participant_id) REFERENCES participants(participant_id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learner_info (
                participant_id INTEGER,
                learning_institution TEXT,
                learning_duration TEXT,
                current_level TEXT,
                formal_training TEXT,
                training_center TEXT,
                
                FOREIGN KEY(participant_id) REFERENCES participants(participant_id)
            )
        ''')


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
    #return [(int(row['id']), float(row['a']), float(row['b']), float(row['c'])) for row in rows]
    return [(float(row['a']), float(row['b']), float(row['c'])) for row in rows]

# ------------------ توابع IRT ------------------

def three_pl_probability(theta, a, b, c):
    return c + (1 - c) / (1 + np.exp(-a * (theta - b)))

def estimate_theta_mle(responses, item_params, lr=0.005, max_iter=500, tol=1e-5):
    theta = 0.0
    for _ in range(max_iter):
        grad = 0.0
        for i, (a, b, c) in enumerate(item_params):
            p = three_pl_probability(theta, a, b, c)
            q = 1 - p
            # اصلاح گرادیان ساده‌تر و ایمن‌تر
            numerator = (responses[i] - p) * a * (1 - c)
            denominator = p * q + 1e-9
            dL = numerator / denominator
            grad += dL
        theta_new = theta + lr * grad
        # محدود کردن θ در دامنه [-4, 4]
        theta_new = max(min(theta_new, 4), -4)
        if abs(theta_new - theta) < tol:
            break
        theta = theta_new
    return float(theta)



def estimate_theta_map(responses, item_params, lr=0.01, max_iter=500, tol=1e-5):
    theta = 0.0
    prior_mean = 0.0
    prior_var = 1.0  # واریانس prior، می‌توانید مقدارش را تغییر دهید
    for _ in range(max_iter):
        grad = 0.0
        for i, (a, b, c) in enumerate(item_params):
            p = three_pl_probability(theta, a, b, c)
            q = 1 - p
            dL = a * (responses[i] - p) * (1 - c) / (p * q + 1e-9)
            grad += dL
        # گرادیان prior (نرمال با میانگین صفر و واریانس prior_var)
        grad_prior = -(theta - prior_mean) / prior_var
        grad_total = grad + grad_prior
        theta_new = theta + lr * grad_total
        if abs(theta_new - theta) < tol:
            break
        theta = theta_new
    return float(theta)


def item_information(theta, a, b, c):
    p = three_pl_probability(theta, a, b, c)
    q = 1 - p
    denominator = ((1 - c) ** 2 * p * q) + 1e-9
    if p <= 0 or p >= 1 or np.isnan(p) or np.isnan(q) or denominator <= 1e-9:
        return -np.inf  # اطلاعات بی‌ارزش یا محاسبه‌ناپذیر
    info = (a ** 2) * ((p - c) ** 2) / denominator
    return info if not np.isnan(info) and info > 1e-6 else -np.inf


def select_next_question(theta, all_item_params, answered_indices):
    infos = []
    for i, (a, b, c) in enumerate(all_item_params):
        if i in answered_indices:
            infos.append(-np.inf)
        else:
            info = item_information(theta, a, b, c)
            infos.append(info)

    print("اطلاعات سوالات:", infos)

    if all(info == -np.inf for info in infos):
        print("❌ هیچ سوال قابل انتخابی وجود ندارد.")
        return None

    next_q = int(np.argmax(infos))
    return next_q


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

def get_correct_answer(question_id):
    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute("SELECT correct_option FROM questions WHERE id = ?", (question_id,))
    result = cursor.fetchone()
    return result['correct_option'] if result else None

@app.route('/')
def index():
    session.clear()
    session['answered_questions'] = []
    session['responses'] = []
    session['theta'] = 0.0
    
    return redirect(url_for('register'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            # اطلاعات عمومی همه شرکت‌کنندگان
            name = request.form.get('name')
            nationality = request.form.get('nationality')
            mother_tongue = request.form.get('mother_tongue')
            official_language = request.form.get('official_language')
            age = request.form.get('age')
            major = request.form.get('major')
            education_level = request.form.get('education_level')
            job = request.form.get('job')
            role = request.form.get('role')

            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()

            # درج در جدول participants
            cursor.execute("""
                INSERT INTO participants 
                (name, nationality, mother_tongue, official_language, age, major, education_level, job, role)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (name, nationality, mother_tongue, official_language, age, major, education_level, job, role))

            participant_id = cursor.lastrowid  # دریافت id برای درج در جدول‌های مرتبط

            if role == 'teacher':
                teaching_experience = request.form.get('teaching_years')
                institutions = request.form.get('teaching_institutions')
                teaching_level = request.form.get('teaching_level')
                academic_persian_opinion = request.form.get('academic_persian_opinion')

                cursor.execute("""
                    INSERT INTO teacher_info 
                    (participant_id, teaching_experience, institutions, teaching_level, importance_of_academic_persian)
                    VALUES (?, ?, ?, ?, ?)
                """, (participant_id, teaching_experience, institutions, teaching_level, academic_persian_opinion))

            elif role == 'learner':
                learning_duration = request.form.get('learning_duration')
                current_persian_level = request.form.get('current_level')
                formal_training = request.form.get('formal_training')
                training_institution = request.form.get('training_institution')
                samfa_taken = request.form.get('samfa_taken')
                samfa_score = request.form.get('samfa_score')
                writing_ability = request.form.get('writing_ability')
                reading_ability = request.form.get('reading_ability')
                speaking_ability = request.form.get('speaking_ability')
                listening_ability = request.form.get('listening_ability')
                importance_of_academic_persian = request.form.get('importance_of_academic_persian')

                cursor.execute("""
                INSERT INTO learner_info 
                (participant_id, learning_duration, current_persian_level, formal_training, 
                training_institution, samfa_taken, samfa_score, importance_of_academic_persian,
                speaking_ability,reading_ability ,writing_ability,listening_ability)
                VALUES (?, ?, ?, ?, ?, ?, ?, ? ,? ,? ,? ,?) 
            """, (participant_id, learning_duration, current_persian_level, formal_training,
                training_institution, samfa_taken, samfa_score, importance_of_academic_persian, speaking_ability, 
                reading_ability, writing_ability, listening_ability))

            flash('موفق شدم!!!!!!!!!!!!!!', 'ok')
            conn.commit()
            conn.close()

            session['participant_id'] = participant_id
            session['user_name'] = name
            # می‌توان به صفحه شروع آزمون هدایت کرد یا پیام موفقیت داد
            return redirect(url_for('test'))  # فرض بر این است که چنین روتی دارید

        except sqlite3.Error as e:
            # در صورت خطای پایگاه داده، آن را چاپ کنید
            print("خطای پایگاه داده:", e)
            flash('موفق شدم!!!!!!!!!!!!!!', 'ok')

            conn.rollback()  # در صورت خطا، تغییرات را برگردانید
            return f"خطای پایگاه داده: {e}", 500

        except Exception as e:
            print("خطای عمومی:", e)

            return f"خطا: {e}", 500
    return render_template('register.html')


@app.route('/test', methods=['GET', 'POST'])
def test():
    
    # در ابتدای تابع
    all_item_params_raw = get_all_item_params()
    question_ids = [item[0] for item in all_item_params_raw]
    all_item_params = [item[1:] for item in all_item_params_raw]  # فقط a, b, c


    if 'participant_id' not in session:
        return redirect(url_for('index'))

    # اگر جواب سوالات قبلی وجود ندارد، شروع مجدد به index
    if 'answered_questions' not in session:
        return redirect(url_for('index'))

    answered = list(map(int, session.get('answered_questions', [])))
    responses = list(map(int, session.get('responses', [])))
    theta = float(session.get('theta', 0.0))

    all_item_params = get_all_item_params()
    total_questions = len(all_item_params)

    MIN_QUESTIONS = 8
    MAX_QUESTIONS = 30
    THETA_CHANGE_THRESHOLD = 0.05

    if request.method == 'POST':
        selected_option = request.form.get('answer')
        if selected_option is None:
            current_q_index = answered[-1] if answered else 0
            question = get_question_by_id(current_q_index + 1)
            progress = int(len(answered) / total_questions * 100)
            return render_template('test.html', question=question, error="لطفا یک گزینه را انتخاب کنید.", progress=progress)

        # سوال فعلی (آخرین سوال جواب داده شده)
        current_question_id = answered[-1] 
        correct_option = get_correct_answer(current_question_id)  # گزینه صحیح سوال
        is_correct = int(selected_option) == correct_option
        responses.append(1 if is_correct else 0)

        # تخمین تتا با پاسخ‌های موجود و پارامتر سوالات جواب داده شده
        answered_params = [all_item_params[i] for i in answered]
        old_theta = theta
        theta = estimate_theta_mle(responses, answered_params)
        theta_change = abs(theta - old_theta)
        print(f"θ تغییر: {theta_change:.4f}, آستانه: {THETA_CHANGE_THRESHOLD}")

        # ذخیره پاسخ در دیتابیس
        db = get_db_connection()
        cursor = db.cursor()
        participant_id = session['participant_id']
        cursor.execute(
            "INSERT INTO answers (user_id, question_id, response) VALUES (?, ?, ?)",
            (participant_id, current_question_id, int(selected_option))
        )
        db.commit()

        num_answered = len(answered) + 1  # چون الان پاسخ جدید هم اضافه شده

        # شرط پایان آزمون اصلاح شده (بدون +1 اضافی)
        if (num_answered >= 20 and theta_change < THETA_CHANGE_THRESHOLD) or num_answered >= MAX_QUESTIONS:
            print("آزمون پایان یافت - شرط پایان برقرار است.")
            cursor.execute("SELECT id FROM user_results WHERE user_id = ?", (participant_id,))
            existing = cursor.fetchone()
            if existing:
                cursor.execute("UPDATE user_results SET theta = ? WHERE user_id = ?", (theta, participant_id))
            else:
                cursor.execute("INSERT INTO user_results (user_id, theta) VALUES (?, ?)", (participant_id, theta))
            db.commit()

            # ذخیره وضعیت نهایی در سشن
            session['theta'] = float(theta)
            session['answered_questions'] = list(map(int, answered))
            session['responses'] = list(map(int, responses))
            return redirect(url_for('result'))

        # انتخاب سوال بعدی
        next_q = select_next_question(theta, all_item_params, answered)
        if next_q is None:
            print("هیچ سوال جدیدی برای انتخاب وجود ندارد، انتقال به نتیجه.")
            session['theta'] = float(theta)
            session['answered_questions'] = list(map(int, answered))
            session['responses'] = list(map(int, responses))
            return redirect(url_for('result'))

        # افزودن سوال بعدی به لیست جواب داده شده
        answered.append(next_q)

        # به‌روزرسانی سشن
        session['answered_questions'] = list(map(int, answered))
        session['responses'] = list(map(int, responses))
        session['theta'] = float(theta)
        session.modified = True

        question = get_question_by_id(next_q) # تغییر دادم +1
        progress = int(len(answered) / total_questions * 100)
        return render_template('test.html', question=question, progress=progress)

    # حالت GET برای اولین بار ورود به آزمون
    if not answered:
        next_q = 0
        answered = [next_q]
        session['answered_questions'] = answered
        session['responses'] = []
        session['theta'] = 0.0
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
    if not os.path.exists(DATABASE):
        init_db()
    app.run(debug=True)

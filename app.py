from flask import Flask, render_template, request, redirect, session, send_file
import sqlite3
import irt
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # حتما کلید امن انتخاب کن

DB_PATH = 'questions.db'  # مسیر فایل دیتابیس (اگر متفاوت است تغییر بده)

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

# صفحه ثبت نام و شروع آزمون
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        name = request.form.get('name')
        if not name:
            return render_template('index.html', error="لطفا نام خود را وارد کنید.")
        # ذخیره شرکت‌کننده و ذخیره در session
        participant_id = save_participant(name)
        session['participant_id'] = participant_id
        session['name'] = name
        session['theta'] = 0.0
        session['responses'] = []
        session['answered_questions'] = []
        
        # گرفتن سوالات
        questions = get_all_questions()
        session['questions'] = [dict(q) for q in questions]  # تبدیل Row به dict برای JSON سازی
        
        # انتخاب سوال اول (مثلاً کمترین b که هنوز پاسخ داده نشده)
        next_q = select_next_question(session['theta'], session['questions'], session['answered_questions'])
        print(f"اولین سوال انتخاب شده: {next_q}")
        
        if next_q is None:
            return render_template('index.html', error="هیچ سوالی برای شروع آزمون یافت نشد.")
        
        session['current_question'] = next_q
        return redirect('/test')
    
    # این خط باید دقیقا در سطح بیرونی if باشد
    return render_template('index.html')


# تابع انتخاب سوال بعدی
def select_next_question(theta, questions, answered_ids):
    # سوالاتی که پاسخ داده نشده اند را فیلتر کن
    remaining = [q for q in questions if q['id'] not in answered_ids]
    print(f"تعداد سوالات باقی‌مانده: {len(remaining)}")
    if not remaining:
        return None
    # پیدا کردن سوال با بیشترین اطلاعات در نقطه θ
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
    
    if request.method == 'POST':
        # دریافت پاسخ کاربر
        answer = request.form.get('answer')
        if answer is None or answer not in ['0', '1']:
            return render_template('test.html', question=session['current_question'], error="لطفا پاسخ درست را انتخاب کنید.")
        
        answer_int = int(answer)
        curr_q = session['current_question']
        
        # ذخیره پاسخ در دیتابیس
        save_answer(session['participant_id'], curr_q['id'], answer_int)
        
        # آپدیت پاسخ‌ها و سوالات جواب داده شده در session
        session['responses'].append(answer_int)
        session['answered_questions'].append(curr_q['id'])
        
        # پارامترهای سوالات جواب داده شده
        answered_params = get_item_params_by_ids(session['answered_questions'], session['questions'])
        
        # تخمین θ جدید
        new_theta = irt.estimate_theta_mle(session['responses'], answered_params)
        session['theta'] = new_theta
        
        # شرط توقف (مثال: حد اکثر 15 سوال)
        if len(session['answered_questions']) >= 15:
            return redirect('/result')
        
        # انتخاب سوال بعدی
        next_q = select_next_question(session['theta'], session['questions'], session['answered_questions'])
        if next_q is None:
            return render_template('index.html', error="هیچ سوالی برای شروع آزمون یافت نشد.")
        
        session['current_question'] = next_q
        return redirect('/test')
    
    # نمایش سوال فعلی
    return render_template('test.html', question=session.get('current_question'), theta=session.get('theta'))

# صفحه نمایش نتیجه
@app.route('/result')
def result():
    if 'participant_id' not in session:
        return redirect('/')
    # ذخیره نتایج در فایل ورد و اکسل
    answered_params = get_item_params_by_ids(session['answered_questions'], session['questions'])
    word_path = f"results/result_{session['participant_id']}.docx"
    excel_path = f"results/result_{session['participant_id']}.xlsx"
    
    # اطمینان از وجود پوشه
    os.makedirs('results', exist_ok=True)
    
    irt.save_results_to_word(word_path, session['responses'], answered_params, session['theta'])
    irt.save_results_to_excel(excel_path, session['responses'], answered_params, session['theta'])
    
    return render_template('result.html', theta=session['theta'], word_file=word_path, excel_file=excel_path)

# دانلود فایل ورد
@app.route('/download/word')
def download_word():
    if 'participant_id' not in session:
        return redirect('/')
    word_path = f"results/result_{session['participant_id']}.docx"
    return send_file(word_path, as_attachment=True)

# دانلود فایل اکسل
@app.route('/download/excel')
def download_excel():
    if 'participant_id' not in session:
        return redirect('/')
    excel_path = f"results/result_{session['participant_id']}.xlsx"
    return send_file(excel_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)

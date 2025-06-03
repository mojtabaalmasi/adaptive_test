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
        'correct': int(row[9])  # مثلاً 1 یعنی گزینه اول صحیح است
    } for row in rows]
    return questions


def select_next_question(theta, questions, asked_ids):
    candidates = [q for q in questions if q['id'] not in asked_ids]
    if not candidates:
        return None
    infos = [item_information(theta, q['a'], q['b'], q['c']) for q in candidates]
    max_idx = np.argmax(infos)
    return candidates[max_idx]

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

def save_results_to_excel(file_path, responses, item_params, theta):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "نتایج آزمون"

    ws['A1'] = 'شماره سوال'
    ws['B1'] = 'پاسخ داده شده'
    ws['C1'] = 'a'
    ws['D1'] = 'b'
    ws['E1'] = 'c'

    bold_font = Font(bold=True)
    for cell in ['A1','B1','C1','D1','E1']:
        ws[cell].font = bold_font

    for i, (resp, (a, b, c)) in enumerate(zip(responses, item_params), start=1):
        ws.cell(row=i+1, column=1, value=i)
        ws.cell(row=i+1, column=2, value=resp)
        ws.cell(row=i+1, column=3, value=a)
        ws.cell(row=i+1, column=4, value=b)
        ws.cell(row=i+1, column=5, value=c)

    ws.cell(row=len(responses)+3, column=1, value="توانایی تخمینی θ:")
    ws.cell(row=len(responses)+3, column=2, value=theta)

    wb.save(file_path)
    
def generate_performance_report(responses, item_params, theta):
    easy_correct = 0
    hard_wrong = 0
    total_easy = 0
    total_hard = 0

    for resp, (a, b, c) in zip(responses, item_params):
        if b < 0:
            total_easy += 1
            if resp == 1:
                easy_correct += 1
        elif b >= 0:
            total_hard += 1
            if resp == 0:
                hard_wrong += 1

    level = ''
    if theta > 1:
        level = 'بسیار بالا'
    elif theta > 0.5:
        level = 'بالا'
    elif theta > -0.5:
        level = 'متوسط'
    elif theta > -1.5:
        level = 'پایین'
    else:
        level = 'خیلی پایین'

    text = (
        f"بر اساس مدل سه‌پارامتری IRT و پاسخ‌های ثبت‌شده، توانایی تخمینی آزمون‌دهنده برابر با θ = {theta:.3f} است، "
        f"که نشان‌دهنده‌ی سطح توانایی {level} در مقایسه با میانگین گروه مرجع می‌باشد.\n\n"
        f"آزمون‌دهنده به {easy_correct} سؤال از {total_easy} سؤال ساده (دارای دشواری منفی) پاسخ صحیح داده است، "
        f"که بیانگر توانایی نسبی در حل آیتم‌های آسان است.\n"
        f"همچنین آزمون‌دهنده در {hard_wrong} سؤال از {total_hard} سؤال دشوار (دارای دشواری مثبت) پاسخ اشتباه داده است، "
        f"که طبیعی و مطابق انتظار برای سطح توانایی فعلی او می‌باشد.\n\n"
        "با توجه به این اطلاعات، پیشنهاد می‌شود آزمون‌دهنده تمرکز خود را بر سؤالات با دشواری متوسط تا بالا افزایش دهد "
        "و برای بهبود عملکرد از روش‌های تمرینی هدفمند بهره بگیرد."
    )
    return text


def save_results_to_word(file_path, responses, item_params, theta):
    doc = Document()
    doc.add_heading('گزارش نتایج آزمون', 0)

    # جدول پاسخ‌ها و پارامترها
    table = doc.add_table(rows=1, cols=5)
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'شماره سوال'
    hdr_cells[1].text = 'پاسخ داده شده'
    hdr_cells[2].text = 'a'
    hdr_cells[3].text = 'b'
    hdr_cells[4].text = 'c'

    for i, (resp, (a, b, c)) in enumerate(zip(responses, item_params), start=1):
        row_cells = table.add_row().cells
        row_cells[0].text = str(i)
        row_cells[1].text = str(resp)
        row_cells[2].text = f"{a:.3f}"
        row_cells[3].text = f"{b:.3f}"
        row_cells[4].text = f"{c:.3f}"

    doc.add_paragraph(f"\nتوانایی تخمینی θ: {theta:.3f}")

    # تحلیل عملکرد آزمون‌دهنده
    doc.add_heading('تحلیل عملکرد آزمون‌دهنده', level=1)

    analysis = generate_performance_report(responses, item_params, theta)
    doc.add_paragraph(analysis)

    doc.save(file_path)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test', methods=['GET', 'POST'])
def test():
    try:
        questions = load_questions()

        if request.method == 'POST' and 'name' in request.form:
            # شروع آزمون با دریافت نام و اطلاعات اولیه
            session['name'] = request.form['name']
            session['phone'] = request.form['phone']
            session['major'] = request.form['major']
            session['responses'] = []
            session['asked_questions'] = []
            session['theta'] = 0.0

            first_q = select_next_question(0.0, questions, [])
            if first_q is None:
                return "هیچ سوالی برای آزمون وجود ندارد."
            session['current_question'] = first_q
            return render_template('test.html', questions=[first_q], step_number=1, total_steps='نامشخص')

        elif request.method == 'POST':
            current_q = session.get('current_question', None)
            if current_q is None:
                return redirect(url_for('index'))

            ans = request.form.get(f'q{current_q["id"]}')
            if ans is None:
                return "لطفا یک پاسخ انتخاب کنید."

            # دریافت پاسخ و به‌روزرسانی داده‌ها
            responses = session.get('responses', [])
            asked = session.get('asked_questions', [])
            theta = session.get('theta', 0.0)

            # ذخیره پاسخ (به صورت عددی)
            responses.append(int(ans))
            asked.append(current_q['id'])

            # پارامترهای سوالات پرسیده شده
            item_params = [(q['a'], q['b'], q['c']) for q in questions if q['id'] in asked]

            # برآورد توانایی با MLE
            theta = estimate_theta_mle(responses, item_params)

            # ذخیره در سشن
            session['theta'] = theta
            session['responses'] = responses
            session['asked_questions'] = asked

            # انتخاب سوال بعدی
            next_q = select_next_question(theta, questions, asked)
            if next_q is None:
                session['current_question'] = None

                # ذخیره نمودارها و فایل‌ها
                plot_icc(item_params, os.path.join(OUTPUT_FOLDER, 'icc.png'))
                plot_item_information(item_params, os.path.join(OUTPUT_FOLDER, 'item_info.png'))
                save_results_to_excel(os.path.join(OUTPUT_FOLDER, 'results.xlsx'), responses, item_params, theta)
                save_results_to_word(os.path.join(OUTPUT_FOLDER, 'results.docx'), responses, item_params, theta)

                return redirect(url_for('results'))

            session['current_question'] = next_q
            step_number = len(asked) + 1
            return render_template('test.html', questions=[next_q], step_number=step_number, total_steps='نامشخص')

        # اگر روش درخواست GET بود یا حالت دیگری داشت به صفحه اصلی بازگرد
        return redirect(url_for('index'))
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())  # چاپ خطای کامل در کنسول
        return f"خطا در سرور: {str(e)}"


@app.route('/results')
def results():
    theta = session.get('theta', None)
    if theta is None:
        return redirect(url_for('index'))

    return render_template(
        'results.html',
        theta=theta,
        icc_image=url_for('static', filename='icc.png'),
        info_image=url_for('static', filename='item_info.png'),
        excel_file=url_for('static', filename='results.xlsx'),
        word_file=url_for('static', filename='results.docx'),
    )

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)

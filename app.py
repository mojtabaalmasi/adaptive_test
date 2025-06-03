from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
import os
import sqlite3

from irt import (
    estimate_theta_mle,
    plot_icc,
    plot_item_information,
    save_results_to_excel,
    save_results_to_word
)
import logging
logging.basicConfig(level=logging.DEBUG)


app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

DB_PATH = 'questions.db'
OUTPUT_FOLDER = os.path.join(os.getcwd(), 'static')
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def load_questions():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, text, a, b, c FROM questions")
    rows = cursor.fetchall()
    conn.close()
    questions = [{'id': row[0], 'text': row[1], 'a': row[2], 'b': row[3], 'c': row[4]} for row in rows]
    return questions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test', methods=['GET', 'POST'])
def test():
    questions = load_questions()

    # مرحله ورود اطلاعات اولیه کاربر
    if request.method == 'POST' and 'name' in request.form:
        session['name'] = request.form['name']
        session['phone'] = request.form['phone']
        session['major'] = request.form['major']
        session['responses'] = []
        session['asked_questions'] = []
        session['question_index'] = 0
        # نمایش اولین سوال
        return render_template('test.html', questions=[questions[0]], step_number=1, total_steps=len(questions))

    # مرحله پاسخگویی به سوالات
    elif request.method == 'POST':
        responses = session.get('responses', [])
        asked = session.get('asked_questions', [])
        index = session.get('question_index', 0)

        # دریافت id سوال فعلی
        current_question = questions[index]
        q_id = current_question['id']

        # دریافت پاسخ کاربر به سوال فعلی
        ans = request.form.get(f'q{q_id}')
        if ans not in ['0', '1']:
            # اگر پاسخ داده نشده یا اشتباه است، دوباره سوال را نشان بده
            return render_template('test.html', questions=[current_question], step_number=index+1, total_steps=len(questions), error="لطفا به سوال پاسخ دهید.")

        responses.append(int(ans))
        asked.append(q_id)

        index += 1  # رفتن به سوال بعدی

        # به‌روزرسانی session
        session['responses'] = responses
        session['asked_questions'] = asked
        session['question_index'] = index

        if index < len(questions):
            # اگر هنوز سوال باقی است، سوال بعدی را نشان بده
            return render_template('test.html', questions=[questions[index]], step_number=index+1, total_steps=len(questions))
        else:
            # اگر سوالات تمام شدند، محاسبات و نمودارها را انجام بده و به صفحه نتایج برو
            item_params = [(q['a'], q['b'], q['c']) for q in questions]
            theta = estimate_theta_mle(responses, item_params)
            session['theta'] = theta

            plot_icc(item_params, save_path=os.path.join(OUTPUT_FOLDER, 'icc.png'))
            plot_item_information(item_params, save_path=os.path.join(OUTPUT_FOLDER, 'item_info.png'))
            save_results_to_excel(os.path.join(OUTPUT_FOLDER, 'results.xlsx'), responses, item_params, theta)
            save_results_to_word(os.path.join(OUTPUT_FOLDER, 'results.docx'), responses, item_params, theta)

            return redirect(url_for('results'))

    # اگر کاربر مستقیم به این آدرس با GET آمد، برگرد به صفحه اصلی
    return redirect(url_for('index'))

@app.route('/results')
def results():
    try:
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
    except Exception as e:
        return f"Error in results route: {e}"

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)

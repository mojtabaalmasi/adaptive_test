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

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

DB_PATH = 'questions.db'
OUTPUT_FOLDER = os.path.join(os.getcwd(), 'static')
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# خواندن سؤال‌ها و پارامترهای IRT از دیتابیس
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
    questions = load_questions()
    return render_template('index.html', num_questions=len(questions))

@app.route('/test', methods=['GET', 'POST'])
def test():
    questions = load_questions()

    iif request.method == 'POST' and 'name' in request.form:
    session['name'] = request.form['name']
    session['phone'] = request.form['phone']
    session['major'] = request.form['major']
    return render_template('test.html', questions=questions)


    elif request.method == 'POST':
        # مرحله پاسخ‌دهی به سوالات
        responses = []
        for q in questions:
            ans = request.form.get(f"q{q['id']}")
            responses.append(1 if ans == '1' else 0)

        item_params = [(q['a'], q['b'], q['c']) for q in questions]
        theta = estimate_theta_mle(responses, item_params)

        session['responses'] = responses
        session['theta'] = theta

        plot_icc(item_params, save_path=os.path.join(OUTPUT_FOLDER, 'icc.png'))
        plot_item_information(item_params, save_path=os.path.join(OUTPUT_FOLDER, 'item_info.png'))
        save_results_to_excel(os.path.join(OUTPUT_FOLDER, 'results.xlsx'), responses, item_params, theta)
        save_results_to_word(os.path.join(OUTPUT_FOLDER, 'results.docx'), responses, item_params, theta)

        return redirect(url_for('results'))

    return redirect(url_for('index'))


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

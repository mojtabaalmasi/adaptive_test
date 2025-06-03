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

    if request.method == 'POST' and 'name' in request.form:
        session['name'] = request.form['name']
        session['phone'] = request.form['phone']
        session['major'] = request.form['major']
        session['responses'] = []
        session['asked_questions'] = []
        session['question_index'] = 0
        return render_template('test.html', questions=[questions[0]], step_number=1, total_steps='?')

    elif request.method == 'POST':
        responses = session.get('responses', [])
        asked = session.get('asked_questions', [])
        index = session.get('question_index', 0)

        q_id = load_questions()[index]['id']
        ans = request.form.get(f'q{q_id}')
        responses.append(int(ans) if ans == '1' else 0)
        asked.append(q_id)

        index += 1
        if index < len(questions):
            session['responses'] = responses
            session['asked_questions'] = asked
            session['question_index'] = index
            return render_template('test.html', questions=[questions[index]], step_number=index+1, total_steps=len(questions))
        else:
            item_params = [(q['a'], q['b'], q['c']) for q in questions]
            theta = estimate_theta_mle(responses, item_params)
            session['theta'] = theta

            plot_icc(item_params, save_path=os.path.join(OUTPUT_FOLDER, 'icc.png'))
            plot_item_information(item_params, save_path=os.path.join(OUTPUT_FOLDER, 'item_info.png'))
            save_results_to_excel(os.path.join(OUTPUT_FOLDER, 'results.xlsx'), responses, item_params, theta)
            save_results_to_word(os.path.join(OUTPUT_FOLDER, 'results.docx'), responses, item_params, theta)

            return redirect(url_for('results'))

    return redirect(url_for('index'))

@@app.route('/results')
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

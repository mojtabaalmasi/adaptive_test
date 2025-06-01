from flask import Flask, render_template, request, session, redirect, url_for
from irt import irt_1pl, estimate_theta, fisher_information
import sqlite3
import math

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

DATABASE = "questions.db"

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def get_questions():
    conn = get_db_connection()
    questions = conn.execute("SELECT * FROM questions").fetchall()
    conn.close()
    return questions

@app.route("/")
def index():
    session.clear()
    return render_template("index.html")

@app.route("/start")
def start():
    session.clear()
    session['theta'] = 0.0
    session['answers'] = {}
    session['asked'] = []
    return redirect(url_for("next_question"))

@app.route("/question", methods=["GET", "POST"])
def next_question():
    if 'theta' not in session:
        return redirect(url_for("index"))

    theta = session['theta']
    answers = session['answers']
    asked = session['asked']

    conn = get_db_connection()
    questions = conn.execute("SELECT * FROM questions").fetchall()
    conn.close()

    # سوالاتی که هنوز پرسیده نشده
    remaining_questions = [q for q in questions if q['id'] not in asked]

    if request.method == "POST":
        selected = request.form.get("answer")
        qid = int(request.form.get("question_id"))

        # ذخیره پاسخ
        answers[qid] = selected
        asked.append(qid)

        # به‌روزرسانی برآورد توانایی با روش ساده MLE مدل 1PL
        # برای مدل دقیق تر می‌توان توابع پیچیده‌تر را استفاده کرد
        theta = estimate_theta(answers, questions)
        session['theta'] = theta
        session['answers'] = answers
        session['asked'] = asked

    if len(remaining_questions) == 0:
        return redirect(url_for("result"))

    # انتخاب سوال بعدی بر اساس بیشترین اطلاعات (Fisher Information)
    max_info = -1
    next_q = None
    for q in remaining_questions:
        info = fisher_information(theta, q['a'], q['b'])
        if info > max_info:
            max_info = info
            next_q = q

    return render_template("test.html", question=next_q, theta=theta)

@app.route("/result")
def result():
    if 'theta' not in session:
        return redirect(url_for("index"))

    theta = session['theta']
    answers = session.get('answers', {})
    score = 0
    total = len(answers)

    conn = get_db_connection()
    questions = conn.execute("SELECT * FROM questions").fetchall()
    conn.close()

    for q in questions:
        qid = q['id']
        if qid in answers:
            if answers[qid] == q['correct_option']:
                score += 1

    return render_template("result.html", score=score, total=total, theta=theta)

if __name__ == "__main__":
    app.run(debug=True)
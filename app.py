# app.py
from flask import Flask, flash, render_template, request, redirect, session, send_file, url_for
import sqlite3
import numpy as np
import pandas as pd
from docx import Document
import matplotlib.pyplot as plt
import os
import uuid
from werkzeug.utils import secure_filename
from flask import send_from_directory


# ----------------------------- پیکربندی پایه -----------------------------
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret')

DATABASE = os.environ.get('DATABASE_PATH', '/var/data/questions.db')


def get_db_connection():
    conn = sqlite3.connect(DATABASE, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    return conn

# آپلود ویس
VOICE_BASE = os.environ.get('VOICE_PATH', '/var/data/voices')
os.makedirs(VOICE_BASE, exist_ok=True)

app.config['UPLOAD_FOLDER'] = VOICE_BASE
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024  # 15MB

MIME_EXT = {
    'audio/webm': 'webm',
    'video/webm': 'webm',
    'audio/ogg':  'ogg',
    'audio/mp4':  'm4a',   # Safari iOS
    'audio/mpeg': 'mp3',
    'audio/3gpp': '3gp',
    'audio/wav':  'wav'
}
ALLOWED_MIME = set(MIME_EXT.keys())

# فلگ سازگاری: آیا جدول answers ستونی به نام response دارد؟
_ANSWERS_HAS_RESPONSE_COL = None

def answers_has_response_column():
    global _ANSWERS_HAS_RESPONSE_COL
    if _ANSWERS_HAS_RESPONSE_COL is not None:
        return _ANSWERS_HAS_RESPONSE_COL
    with get_db_connection() as conn:
        cols = conn.execute("PRAGMA table_info(answers)").fetchall()
        names = {c['name'] for c in cols}
        _ANSWERS_HAS_RESPONSE_COL = ('response' in names)
    return _ANSWERS_HAS_RESPONSE_COL

# ----------------------------- شِمای جداول (ایجاد در صورت نبود) -----------------------------
def init_db():
    with sqlite3.connect(DATABASE, timeout=30) as conn:
        cur = conn.cursor()
        cur.executescript("""
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS participants (
            participant_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name             TEXT,
            nationality      TEXT,
            mother_tongue    TEXT,
            official_language TEXT,
            age              INTEGER,
            major            TEXT,
            education_level  TEXT,
            job              TEXT,
            role             TEXT
        );

        CREATE TABLE IF NOT EXISTS teacher_info (
            participant_id               INTEGER UNIQUE,
            teaching_years               INTEGER,
            teaching_institutions        TEXT,
            teaching_level               TEXT,
            importance_of_academic_persian TEXT,
            FOREIGN KEY(participant_id) REFERENCES participants(participant_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS learner_info (
            participant_id         INTEGER UNIQUE,
            learning_duration      TEXT,
            current_level          TEXT,
            formal_training        TEXT,
            training_institution   TEXT,
            samfa_taken            TEXT,
            samfa_score            REAL,
            importance_of_academic_persian TEXT,
            speaking_ability       TEXT,
            reading_ability        TEXT,
            writing_ability        TEXT,
            listening_ability      TEXT,
            FOREIGN KEY(participant_id) REFERENCES participants(participant_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            option1 TEXT, option2 TEXT, option3 TEXT, option4 TEXT,
            correct_option INTEGER,
            a REAL NOT NULL, b REAL NOT NULL, c REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS answers (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL,
            question_id     INTEGER NOT NULL,
            -- ستون legacy (در برخی DBهای شما ممکن است وجود داشته باشد)
            -- response INTEGER,
            selected_option INTEGER,
            is_correct      INTEGER,
            created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id)     REFERENCES participants(participant_id) ON DELETE CASCADE,
            FOREIGN KEY(question_id) REFERENCES questions(id)                ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS user_results (
            user_id INTEGER PRIMARY KEY,
            theta   REAL,
            FOREIGN KEY(user_id) REFERENCES participants(participant_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS manager_questions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            text          TEXT NOT NULL,
            display_order INTEGER NOT NULL DEFAULT 0,
            is_required   INTEGER NOT NULL DEFAULT 1 CHECK (is_required IN (0,1)),
            is_active     INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0,1)),
            created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS voice_answers (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            participant_id INTEGER NOT NULL,
            role           TEXT NOT NULL DEFAULT 'manager',
            question_id    INTEGER NOT NULL,
            file_path      TEXT NOT NULL,
            mime_type      TEXT,
            duration_ms    INTEGER,
            size_bytes     INTEGER,
            created_at     DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(participant_id) REFERENCES participants(participant_id) ON DELETE CASCADE,
            FOREIGN KEY(question_id)   REFERENCES manager_questions(id)        ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT NOT NULL,
            category TEXT,
            frequency INTEGER DEFAULT 0,
            target_role TEXT NOT NULL DEFAULT 'learner'
        );


        -- اطلاعات نقش‌محور مدیر مرکز
        CREATE TABLE IF NOT EXISTS manager_info (
            participant_id INTEGER PRIMARY KEY,
            center_name TEXT,
            center_city TEXT,
            center_type TEXT,
            years_as_manager INTEGER,
            num_teachers INTEGER,
            num_learners INTEGER,
            FOREIGN KEY(participant_id) REFERENCES participants(participant_id) ON DELETE CASCADE
        );

        -- پس‌آزمون مدرسان و مدیران (پرسش‌ها)
        CREATE TABLE IF NOT EXISTS teacher_post_questions (
            id INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            dimension TEXT,
            question_type TEXT DEFAULT 'open',
            scale TEXT,
            display_order INTEGER NOT NULL DEFAULT 0,
            is_active INTEGER NOT NULL DEFAULT 1,
            is_required INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS manager_post_questions (
            id INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            dimension TEXT,
            question_type TEXT DEFAULT 'open',
            scale TEXT,
            display_order INTEGER NOT NULL DEFAULT 0,
            is_active INTEGER NOT NULL DEFAULT 1,
            is_required INTEGER NOT NULL DEFAULT 1
        );

        -- پس‌آزمون مدرسان و مدیران (پاسخ‌ها)
        CREATE TABLE IF NOT EXISTS teacher_post_answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            participant_id INTEGER NOT NULL,
            question_id INTEGER NOT NULL,
            answer_value INTEGER,
            answer_text TEXT,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(participant_id) REFERENCES participants(participant_id) ON DELETE CASCADE,
            FOREIGN KEY(question_id)   REFERENCES teacher_post_questions(id) ON DELETE CASCADE,
            UNIQUE(participant_id, question_id)
        );

        CREATE TABLE IF NOT EXISTS manager_post_answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            participant_id INTEGER NOT NULL,
            question_id INTEGER NOT NULL,
            answer_value INTEGER,
            answer_text TEXT,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(participant_id) REFERENCES participants(participant_id) ON DELETE CASCADE,
            FOREIGN KEY(question_id)   REFERENCES manager_post_questions(id) ON DELETE CASCADE,
            UNIQUE(participant_id, question_id)
        );

        -- لاگ جلسات CAT برای تحلیل‌های پژوهشی
        CREATE TABLE IF NOT EXISTS test_sessions (
            session_id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            role TEXT,
            started_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            ended_at DATETIME,
            stop_reason TEXT,
            items_administered INTEGER DEFAULT 0,
            theta_start REAL,
            theta_final REAL,
            se_final REAL,
            FOREIGN KEY(user_id) REFERENCES participants(participant_id) ON DELETE CASCADE
        );

        -- لاگ هر پاسخ در هر گام CAT (اختیاری اما بسیار مفید)
        CREATE TABLE IF NOT EXISTS answers_meta (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            participant_id INTEGER NOT NULL,
            question_id INTEGER NOT NULL,
            step INTEGER NOT NULL,
            selected_option INTEGER,
            is_correct INTEGER,
            theta_before REAL,
            theta_after REAL,
            se_after REAL,
            info REAL,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES test_sessions(session_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS strategy_answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            participant_id INTEGER NOT NULL,
            strategy_id INTEGER NOT NULL,
            choice INTEGER NOT NULL,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME,
            FOREIGN KEY(participant_id) REFERENCES participants(participant_id) ON DELETE CASCADE,
            FOREIGN KEY(strategy_id) REFERENCES strategies(id) ON DELETE CASCADE,
            UNIQUE(participant_id, strategy_id)
        );
        """)

        # ---- مهاجرت سبک (افزودن ستون‌های جدید در صورت وجود جدول‌های قدیمی) ----
        def _ensure_column(table, col, col_def):
            cols = [r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()]
            if col not in cols:
                cur.execute(f"ALTER TABLE {table} ADD COLUMN {col_def};")

        # strategies
        _ensure_column('strategies', 'frequency', 'frequency INTEGER DEFAULT 0')
        _ensure_column('strategies', 'target_role', "target_role TEXT NOT NULL DEFAULT 'learner'")

        # teacher/manager post questions
        for t in ('teacher_post_questions', 'manager_post_questions'):
            _ensure_column(t, 'dimension', 'dimension TEXT')
            _ensure_column(t, 'question_type', "question_type TEXT DEFAULT 'open'")
            _ensure_column(t, 'scale', 'scale TEXT')
            _ensure_column(t, 'display_order', 'display_order INTEGER NOT NULL DEFAULT 0')
            _ensure_column(t, 'is_active', 'is_active INTEGER NOT NULL DEFAULT 1')
            _ensure_column(t, 'is_required', 'is_required INTEGER NOT NULL DEFAULT 1')

        # teacher/manager post answers
        for t in ('teacher_post_answers', 'manager_post_answers'):
            _ensure_column(t, 'answer_value', 'answer_value INTEGER')
            _ensure_column(t, 'answer_text', 'answer_text TEXT')

        # ایندکس‌های موردنیاز برای UPSERT
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_user_results_user ON user_results(user_id);")
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_voice_answers_user_q ON voice_answers(participant_id, question_id);")
        conn.commit()

# ----------------------------- لایهٔ سؤال/پارامترها -----------------------------
def get_question_by_id(question_id: int):
    conn = get_db_connection()
    row = conn.execute(
        "SELECT id, text, option1, option2, option3, option4 FROM questions WHERE id = ?",
        (question_id,)
    ).fetchone()
    conn.close()
    if row:
        return {
            'id': int(row['id']),
            'text': row['text'],
            'options': [row['option1'], row['option2'], row['option3'], row['option4']]
        }
    return None

def get_all_item_params():
    conn = get_db_connection()
    rows = conn.execute("SELECT id, a, b, c FROM questions ORDER BY id").fetchall()
    conn.close()
    return [(int(r['id']), float(r['a']), float(r['b']), float(r['c'])) for r in rows]

def get_correct_answer(question_id: int):
    with get_db_connection() as db:
        row = db.execute("SELECT correct_option FROM questions WHERE id = ?", (question_id,)).fetchone()
    return int(row['correct_option']) if (row and row['correct_option'] is not None) else None

# ----------------------------- IRT (3PL پایدار) -----------------------------
EPS = 1e-9
THETA_MIN, THETA_MAX = -4.0, 4.0

def _sigmoid(x):
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))

def three_pl_probability(theta, a, b, c):
    a = max(float(a), EPS)
    c = float(np.clip(c, 0.0, 0.999))
    p = c + (1.0 - c) * _sigmoid(a * (theta - b))
    return float(np.clip(p, c + EPS, 1.0 - EPS))

def item_information(theta, a, b, c):
    p = three_pl_probability(theta, a, b, c)
    q = 1.0 - p
    denom = (1.0 - c) ** 2 + EPS
    info = (a ** 2) * ((p - c) ** 2) / denom * (q / p)
    return float(info) if np.isfinite(info) and info > 0 else 0.0

def test_information(theta, item_params):
    return float(sum(item_information(theta, a, b, c) for (a, b, c) in item_params))

def theta_se(theta, item_params):
    I = test_information(theta, item_params)
    return float(1.0 / np.sqrt(max(I, EPS)))

def _grad_loglik_theta(theta, responses, item_params):
    g = 0.0
    for x, (a, b, c) in zip(responses, item_params):
        p = three_pl_probability(theta, a, b, c)
        g += a * (x - p) * (p - c) / ((1.0 - c) * p + EPS)
    return float(g)

def estimate_theta_mle(responses, item_params, lr=0.01, max_iter=50, tol=1e-4):
    theta = 0.0
    for _ in range(max_iter):
        g = _grad_loglik_theta(theta, responses, item_params)
        I = test_information(theta, item_params) + EPS
        step = g / I
        if not np.isfinite(step) or abs(step) > 1.0:
            step = 0.25 * np.tanh(step)
        theta_new = float(np.clip(theta + step, THETA_MIN, THETA_MAX))
        if abs(theta_new - theta) < tol:
            return theta_new
        theta = theta_new
    return theta

def estimate_theta_map(responses, item_params, theta0=0.0, prior_mean=0.0, prior_var=1.0,
                       max_iter=50, tol=1e-4):
    theta = float(np.clip(theta0, THETA_MIN, THETA_MAX))
    inv_var = 1.0 / max(prior_var, EPS)
    for _ in range(max_iter):
        g_like = _grad_loglik_theta(theta, responses, item_params)
        g_prior = -(theta - prior_mean) * inv_var
        g = g_like + g_prior
        I = test_information(theta, item_params) + inv_var + EPS
        step = g / I
        if not np.isfinite(step) or abs(step) > 1.0:
            step = 0.25 * np.tanh(step)
        theta_new = float(np.clip(theta + step, THETA_MIN, THETA_MAX))
        if abs(theta_new - theta) < tol:
            return theta_new
        theta = theta_new
    return theta

def select_next_question(theta, all_item_params, answered_indices):
    best_idx, best_info = None, -1.0
    for i, (a, b, c) in enumerate(all_item_params):
        if i in answered_indices:
            continue
        info = item_information(theta, a, b, c)
        if info > best_info:
            best_info, best_idx = info, i
    return best_idx

# ----------------------------- نمودار -----------------------------
def plot_icc(item_params, save_path):
    theta_range = np.linspace(THETA_MIN, THETA_MAX, 200)
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
    theta_range = np.linspace(THETA_MIN, THETA_MAX, 200)
    total_info = np.zeros_like(theta_range, dtype=float)
    for a, b, c in item_params:
        info = np.array([item_information(t, a, b, c) for t in theta_range], dtype=float)
        total_info += info
    plt.figure(figsize=(8, 5))
    plt.plot(theta_range, total_info)
    plt.xlabel('θ (توانایی)')
    plt.ylabel('اطلاعات آزمون')
    plt.title('تابع اطلاعات کل آزمون')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path

# ----------------------------- خروجی به Excel/Word -----------------------------
def save_results_to_excel(filepath, responses, answered_indices, theta):
    conn = get_db_connection()
    rows = conn.execute("SELECT text, a, b, c FROM questions ORDER BY id").fetchall()
    conn.close()
    data = []
    for i, r in zip(answered_indices, responses):
        row = rows[i]
        data.append({'سوال': row['text'], 'پاسخ (0/1)': r, 'a': row['a'], 'b': row['b'], 'c': row['c']})
    df = pd.DataFrame(data)
    df.loc[len(df)] = ['θ (توانایی)', theta, '', '', '']
    df.to_excel(filepath, index=False)

def save_results_to_word(filepath, responses, answered_indices, theta):
    conn = get_db_connection()
    rows = conn.execute("SELECT text, a, b, c FROM questions ORDER BY id").fetchall()
    conn.close()
    doc = Document()
    doc.add_heading('نتایج آزمون تطبیقی (3PL)', 0)
    doc.add_paragraph(f'مقدار تخمینی θ: {theta:.3f}')
    table = doc.add_table(rows=1, cols=5)
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'سوال'
    hdr_cells[1].text = 'پاسخ (0/1)'
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

# ----------------------------- مسیرهای وب -----------------------------
@app.route('/')
def index():
    # هر بار که وارد صفحهٔ اصلی می‌شوی، سشن قبلی پاک شود
    session.clear()
    return render_template('index.html')

@app.route('/voices/<path:filepath>')
def serve_voice(filepath):
    # فایل‌ها از دیسک پایدار سرو می‌شوند
    return send_from_directory(VOICE_BASE, filepath, as_attachment=False)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            with sqlite3.connect(DATABASE, timeout=30) as conn:
                conn.execute("PRAGMA busy_timeout=30000;")
                cur = conn.cursor()

                name = request.form.get('name')
                nationality = request.form.get('nationality')
                mother_tongue = request.form.get('mother_tongue')
                official_language = request.form.get('official_language')
                age = request.form.get('age')
                major = request.form.get('major')
                education_level = request.form.get('education_level')
                job = request.form.get('job')
                role = request.form.get('role')

                cur.execute("""
                    INSERT INTO participants
                    (name, nationality, mother_tongue, official_language, age, major, education_level, job, role)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (name, nationality, mother_tongue, official_language, age, major, education_level, job, role))
                participant_id = cur.lastrowid

                if role == 'teacher':
                    teaching_years = request.form.get('teaching_years')
                    teaching_institutions = request.form.get('teaching_institutions')
                    teaching_level = request.form.get('teaching_level')
                    academic_persian_opinion = request.form.get('academic_persian_opinion')
                    cur.execute("""
                        INSERT INTO teacher_info
                        (participant_id, teaching_years, teaching_institutions, teaching_level, importance_of_academic_persian)
                        VALUES (?, ?, ?, ?, ?)
                    """, (participant_id, teaching_years, teaching_institutions, teaching_level, academic_persian_opinion))

                elif role == 'learner':
                    learning_duration = request.form.get('learning_duration')
                    current_level = request.form.get('current_level')
                    formal_training = request.form.get('formal_training')
                    training_institution = request.form.get('training_institution')
                    samfa_taken = request.form.get('samfa_taken')
                    samfa_score = request.form.get('samfa_score')
                    writing_ability = request.form.get('writing_ability')
                    reading_ability = request.form.get('reading_ability')
                    speaking_ability = request.form.get('speaking_ability')
                    listening_ability = request.form.get('listening_ability')
                    importance_of_academic_persian = request.form.get('importance_of_academic_persian')
                    cur.execute("""
                        INSERT INTO learner_info
                        (participant_id, learning_duration, current_level, formal_training,
                         training_institution, samfa_taken, samfa_score, importance_of_academic_persian,
                         speaking_ability, reading_ability, writing_ability, listening_ability)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (participant_id, learning_duration, current_level, formal_training,
                          training_institution, samfa_taken, samfa_score, importance_of_academic_persian,
                          speaking_ability, reading_ability, writing_ability, listening_ability))
                elif role == 'manager':
                    # اطلاعات نقش‌محور مدیر مرکز (اختیاری؛ اگر در فرم فیلدی موجود نباشد NULL ذخیره می‌شود)
                    center_name = request.form.get('center_name')
                    center_city = request.form.get('center_city')
                    center_type = request.form.get('center_type')
                    years_as_manager = request.form.get('years_as_manager')
                    num_teachers = request.form.get('num_teachers')
                    num_learners = request.form.get('num_learners')
                    cur.execute("""
                        INSERT INTO manager_info
                        (participant_id, center_name, center_city, center_type, years_as_manager, num_teachers, num_learners)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (participant_id, center_name, center_city, center_type,
                          years_as_manager, num_teachers, num_learners))

                conn.commit()

            session['participant_id'] = participant_id
            session['user_name'] = name
            session['role'] = role
            return redirect(url_for('test'))

        except sqlite3.Error as e:
            print("DB Error:", e)
            return f"خطای پایگاه داده: {e}", 500
        except Exception as e:
            print("Error:", e)
            return f"خطا: {e}", 500
    return render_template('register.html')



# ----------------------------- آزمون تطبیقی -----------------------------
@app.route('/test', methods=['GET', 'POST'])
def test():
    if 'participant_id' not in session:
        return redirect(url_for('index'))

    session.setdefault('answered_questions', [])
    session.setdefault('responses', [])
    session.setdefault('theta', 0.0)
    session.setdefault('stable_streak', 0)

    session.setdefault('test_session_id', None)
    if not session.get('test_session_id'):
        session['test_session_id'] = str(uuid.uuid4())
        # ایجاد رکورد جلسه برای تحلیل پژوهشی
        with sqlite3.connect(DATABASE, timeout=30) as db:
            db.execute("PRAGMA busy_timeout=30000;")
            db.execute(
                "INSERT OR IGNORE INTO test_sessions (session_id, user_id, role, theta_start) VALUES (?, ?, ?, ?)",
                (session['test_session_id'], session['participant_id'], session.get('role'), float(session.get('theta', 0.0)))
            )
            db.commit()


    answered = list(map(int, session['answered_questions']))
    responses = list(map(int, session['responses']))
    theta = float(session['theta'])
    streak = int(session['stable_streak'])

    rows = get_all_item_params()
    if not rows:
        flash('بانک سؤال خالی است.', 'error')
        return redirect(url_for('index'))
    question_ids = [r[0] for r in rows]
    all_item_params = [tuple(r[1:]) for r in rows]
    total_questions = len(all_item_params)

    MIN_QUESTIONS = 8
    HARD_MAX     = 22
    SE_TARGET    = 0.30
    DELTA_TARGET = 0.03
    STREAK_NEED  = 2
    CUT_SCORE    = 0.0
    Z_CI         = 1.96

    if request.method == 'GET' and not answered:
        theta = 0.0
        responses = []
        streak = 0

        start_idx = select_next_question(theta, all_item_params, answered_indices=[])
        if start_idx is None:
            flash('سؤالی برای شروع یافت نشد.', 'error')
            return redirect(url_for('index'))

        answered = [start_idx]
        session['answered_questions'] = answered
        session['responses'] = responses
        session['theta'] = float(theta)
        session['stable_streak'] = int(streak)
        session.modified = True

        current_idx = answered[-1]
        current_qid = question_ids[current_idx]
        question = get_question_by_id(current_qid)
        progress = int(len(answered) / max(total_questions, 1) * 100)
        return render_template('test.html', question=question, progress=progress)

    if request.method == 'POST':
        selected_option = request.form.get('answer')
        try:
            sel = int(selected_option)
        except (TypeError, ValueError):
            sel = None

        if sel not in (1, 2, 3, 4):
            current_idx = answered[-1] if answered else 0
            current_qid = question_ids[current_idx]
            question = get_question_by_id(current_qid)
            progress = int(len(answered) / max(total_questions, 1) * 100)
            return render_template(
                'test.html', question=question,
                error="گزینهٔ معتبر انتخاب نشده است.", progress=progress
            )

        current_idx = answered[-1]
        current_qid = question_ids[current_idx]

        co = get_correct_answer(current_qid)
        try:
            co_int = int(co)
        except (TypeError, ValueError):
            co_int = None
        is_correct = 1 if (co_int is not None and sel == co_int) else 0
        responses.append(is_correct)

        answered_params = [all_item_params[i] for i in answered]
        old_theta = theta
        if len(responses) < 3:
            theta = estimate_theta_map(responses, answered_params, prior_mean=0.0, prior_var=1.0)
        else:
            theta = estimate_theta_mle(responses, answered_params)
        theta_change = abs(theta - old_theta)
        se_now = theta_se(theta, answered_params)

        participant_id = session['participant_id']
        with sqlite3.connect(DATABASE, timeout=30) as db:
            db.execute("PRAGMA busy_timeout=30000;")
            cur = db.cursor()
            try:
                has_resp_col = answers_has_response_column()
            except NameError:
                has_resp_col = False

            if has_resp_col:
                cur.execute(
                    "INSERT INTO answers (user_id, question_id, response, selected_option, is_correct) VALUES (?, ?, ?, ?, ?)",
                    (participant_id, current_qid, sel, sel, is_correct)
                )
            else:
                cur.execute(
                    "INSERT INTO answers (user_id, question_id, selected_option, is_correct) VALUES (?, ?, ?, ?)",
                    (participant_id, current_qid, sel, is_correct)
                )
            # ثبت لاگ گام‌به‌گام برای تحلیل (answers_meta)
            try:
                cur.execute(
                    """
                    INSERT INTO answers_meta
                    (session_id, participant_id, question_id, step, selected_option, is_correct, theta_before, theta_after, se_after)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session.get('test_session_id'),
                        participant_id,
                        current_qid,
                        len(responses),
                        sel,
                        is_correct,
                        float(old_theta),
                        float(theta),
                        float(se_now) if se_now is not None else None
                    )
                )
            except sqlite3.Error:
                # اگر جدول answers_meta در DB وجود نداشت یا ساختارش فرق داشت، آزمون مختل نشود
                pass

            # به‌روزرسانی خلاصه جلسه
            try:
                cur.execute(
                    """
                    UPDATE test_sessions
                    SET items_administered = ?,
                        theta_final = ?,
                        se_final = ?
                    WHERE session_id = ?
                    """,
                    (len(responses), float(theta), float(se_now) if se_now is not None else None, session.get('test_session_id'))
                )
            except sqlite3.Error:
                pass

            db.commit()

        if theta_change < DELTA_TARGET:
            streak += 1
        else:
            streak = 0
        session['stable_streak'] = int(streak)

        num_answered = len(responses)
        above_cut = (se_now is not None) and ((theta - Z_CI * se_now) > CUT_SCORE)
        below_cut = (se_now is not None) and ((theta + Z_CI * se_now) < CUT_SCORE)

        stop_reason = None
        if num_answered >= MIN_QUESTIONS:
            if se_now is not None and se_now <= SE_TARGET:
                stop_reason = f"دقت کافی (SE ≤ {SE_TARGET})"
            elif above_cut:
                stop_reason = f"نتیجهٔ قطعی: بالاتر از مرز {CUT_SCORE}"
            elif below_cut:
                stop_reason = f"نتیجهٔ قطعی: پایین‌تر از مرز {CUT_SCORE}"
            elif streak >= STREAK_NEED:
                stop_reason = f"پایداری θ (Δθ < {DELTA_TARGET} برای {STREAK_NEED} بار پیاپی)"

        if stop_reason is None and num_answered >= HARD_MAX:
            stop_reason = f"رسیدن به سقف {HARD_MAX} سؤال"

        if stop_reason is not None:
            with sqlite3.connect(DATABASE, timeout=30) as conn:
                conn.execute("PRAGMA busy_timeout=30000;")
                cur = conn.cursor()
                cur.execute("UPDATE user_results SET theta=? WHERE user_id=?", (float(theta), participant_id))
                if cur.rowcount == 0:
                    cur.execute("INSERT INTO user_results (user_id, theta) VALUES (?, ?)", (participant_id, float(theta)))
                # بستن جلسه آزمون برای تحلیل
                try:
                    cur.execute(
                        """
                        UPDATE test_sessions
                        SET ended_at = CURRENT_TIMESTAMP,
                            stop_reason = ?,
                            items_administered = ?,
                            theta_final = ?,
                            se_final = ?
                        WHERE session_id = ?
                        """,
                        (
                            stop_reason,
                            len(responses),
                            float(theta),
                            float(se_now) if se_now is not None else None,
                            session.get('test_session_id')
                        )
                    )
                except sqlite3.Error:
                    pass

                conn.commit()

            session['theta'] = float(theta)
            session['answered_questions'] = answered
            session['responses'] = responses
            session['stop_reason'] = stop_reason
            session.modified = True
            return redirect(url_for('result'))

        next_idx = select_next_question(theta, all_item_params, answered)
        if next_idx is None:
            with sqlite3.connect(DATABASE, timeout=30) as conn:
                conn.execute("PRAGMA busy_timeout=30000;")
                cur = conn.cursor()
                cur.execute("UPDATE user_results SET theta=? WHERE user_id=?", (float(theta), participant_id))
                if cur.rowcount == 0:
                    cur.execute("INSERT INTO user_results (user_id, theta) VALUES (?, ?)", (participant_id, float(theta)))
                conn.commit()

            session['theta'] = float(theta)
            session['answered_questions'] = answered
            session['responses'] = responses
            session['stop_reason'] = "پایان بانک سؤال"
            session.modified = True
            return redirect(url_for('result'))

        answered.append(next_idx)
        session['answered_questions'] = answered
        session['responses'] = responses
        session['theta'] = float(theta)
        session.modified = True

        next_qid = question_ids[next_idx]
        question = get_question_by_id(next_qid)
        progress = int(len(answered) / max(total_questions, 1) * 100)
        return render_template('test.html', question=question, progress=progress)

    current_idx = answered[-1]
    current_qid = question_ids[current_idx]
    question = get_question_by_id(current_qid)
    progress = int(len(answered) / max(total_questions, 1) * 100)
    return render_template('test.html', question=question, progress=progress)

# ----------------------------- پس آزمون راهبردها -----------------------------
@app.route('/post_test', methods=['GET', 'POST'])
def post_test():
    if 'participant_id' not in session:
        return redirect(url_for('index'))

    participant_id = int(session['participant_id'])
    role = session.get('role', 'learner')  # 'manager' یا 'teacher' یا 'learner'

    # --- خواندن آیتم‌ها از DB بر اساس نقش ---
    with sqlite3.connect(DATABASE, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000;")
        cur = conn.cursor()
        rows = cur.execute("""
            SELECT id, strategy AS text, COALESCE(category, '') AS category
            FROM strategies
            WHERE target_role = ?
            ORDER BY category, id
        """, (role,)).fetchall()


    if not rows:
        flash("هیچ موردی در پرسشنامهٔ راهبردها تعریف نشده است.", "error")
        return redirect(url_for('result'))

    # ساخت گروه‌ها
    groups = []
    last_cat = None
    for r in rows:
        cat = r["category"]
        if cat != last_cat:
            groups.append({"category": cat, "items": []})
            last_cat = cat
        groups[-1]["items"].append({"id": int(r["id"]), "text": r["text"]})

    if request.method == "POST":
        errors = {}
        payload = {}

        for g in groups:
            for item in g["items"]:
                key = f"s_{item['id']}"
                val = request.form.get(key)

                if not val:
                    errors[item["id"]] = "الزامی"
                    continue

                try:
                    choice = int(val)
                except ValueError:
                    errors[item["id"]] = "نامعتبر"
                    continue

                if not (1 <= choice <= 5):
                    errors[item["id"]] = "بازه ۱ تا ۵"
                    continue

                payload[item["id"]] = choice

        if errors:
            return render_template(
                "strategies_survey.html",
                groups=groups,
                errors=errors,
                values=request.form,
            )

        with sqlite3.connect(DATABASE, timeout=30) as conn:
            conn.execute("PRAGMA busy_timeout=30000;")
            cur = conn.cursor()
            for sid, choice in payload.items():
                cur.execute(
                    """
                    INSERT INTO strategy_answers (participant_id, strategy_id, choice)
                    VALUES (?, ?, ?)
                    ON CONFLICT(participant_id, strategy_id)
                    DO UPDATE SET choice=excluded.choice,
                                  updated_at=CURRENT_TIMESTAMP
                    """,
                    (participant_id, sid, choice),
                )
            conn.commit()

        session["post_test_saved"] = True
        return redirect(url_for("thank_you"))

    return render_template(
        "strategies_survey.html",
        groups=groups,
        errors={},
        values={},
    )

# ----------------------------- قدردانی -----------------------------
@app.route('/thank_you')
def thank_you():
    if 'participant_id' not in session:
        return redirect(url_for('index'))
    return render_template('thank_you.html', user_name=session.get('user_name', 'کاربر گرامی'))

# ----------------------------- نتیجه -----------------------------
@app.route('/result')
def result():
    if 'responses' not in session or 'answered_questions' not in session:
        return redirect(url_for('index'))

    responses = list(map(int, session['responses']))          # 0/1
    answered = list(map(int, session['answered_questions']))  # اندیس‌های 0-مبنا
    theta = float(session.get('theta', 0.0))

    # پارامترهای آیتم‌های پاسخ‌داده‌شده
    rows = get_all_item_params()
    all_item_params = [tuple(r[1:]) for r in rows]
    answered_params = [all_item_params[i] for i in answered] if answered else []

    # نمودارها
    icc_path = plot_icc(answered_params, f"static/icc_{uuid.uuid4().hex}.png") if answered_params else None
    info_path = plot_item_information(answered_params, f"static/info_{uuid.uuid4().hex}.png") if answered_params else None

    # آمار
    n_total = len(responses)
    n_correct = sum(1 for r in responses if r == 1)
    n_wrong = n_total - n_correct
    accuracy = round((n_correct / n_total) * 100, 1) if n_total > 0 else 0

    # SE و بازه‌ها
    if answered_params:
        se = theta_se(theta, answered_params)
        ci68 = (max(-4, theta - se), min(4, theta + se))
        ci95 = (max(-4, theta - 1.96 * se), min(4, theta + 1.96 * se))
    else:
        se, ci68, ci95 = None, None, None

    # باند
    def ability_band(t):
        if t < -2: return "خیلی پایین"
        elif t < -1: return "پایین"
        elif t <= 1: return "متوسط"
        elif t <= 2: return "بالا"
        else: return "خیلی بالا"

    band = ability_band(theta)

    # تفسیر
    if band == "خیلی پایین":
        interpretation = "نتیجه نشان می‌دهد سطح توانایی شما بسیار پایین است..."
    elif band == "پایین":
        interpretation = "سطح توانایی شما پایین‌تر از میانگین است..."
    elif band == "متوسط":
        interpretation = "توانایی شما نزدیک به میانگین است..."
    elif band == "بالا":
        interpretation = "توانایی شما بالاتر از میانگین است..."
    else:
        interpretation = "توانایی شما بسیار بالا برآورد شده است..."

    user_name = session.get('user_name', 'کاربر ناشناس')

    # -------------------------------------------------------------
    #  چک کردن اینکه کدام پس‌آزمون باید نمایش داده شود
    # -------------------------------------------------------------
    role = session.get('role', 'learner')
    pid = int(session['participant_id'])

    post_test_url = None
    post_test_done = False

    with sqlite3.connect(DATABASE, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000;")
        cur = conn.cursor()

        if role == 'learner':
            post_test_url = url_for('post_test')
            row = cur.execute(
                "SELECT 1 FROM strategy_answers WHERE participant_id=? LIMIT 1",
                (pid,)
            ).fetchone()
            post_test_done = (row is not None)

        elif role == 'teacher':
            post_test_url = url_for('post_test_teacher')
            row = cur.execute(
                "SELECT 1 FROM teacher_post_answers WHERE participant_id=? LIMIT 1",
                (pid,)
            ).fetchone()
            post_test_done = (row is not None)

        elif role == 'manager':
            post_test_url = url_for('post_test_manager')
            row = cur.execute(
                "SELECT 1 FROM manager_post_answers WHERE participant_id=? LIMIT 1",
                (pid,)
            ).fetchone()
            post_test_done = (row is not None)

    # has_post_test برای سازگاری با قالب قدیمی
    has_post_test = post_test_done

    return render_template(
        'result.html',
        theta=theta,
        band=band,
        se=se,
        ci68=ci68,
        ci95=ci95,
        n_total=n_total,
        n_correct=n_correct,
        n_wrong=n_wrong,
        accuracy=accuracy,
        user_name=user_name,
        icc_image=icc_path,
        info_image=info_path,
        interpretation=interpretation,
        has_post_test=has_post_test,
        role=role,
        post_test_url=post_test_url,
        post_test_done=post_test_done
    )

##-----------------------------------پس آزمون----------------
@app.route('/post_test_teacher', methods=['GET', 'POST'])
def post_test_teacher():
    if 'participant_id' not in session or session.get('role') != 'teacher':
        return redirect(url_for('index'))

    pid = session['participant_id']

    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # آیا قبلاً پر کرده؟
    row = cur.execute(
        "SELECT 1 FROM teacher_post_answers WHERE participant_id=? LIMIT 1",
        (pid,)
    ).fetchone()

    if row:
        return render_template("post_test_completed.html", role="teacher")

    # خواندن سوالات
    questions = cur.execute(
        "SELECT id, text, COALESCE(question_type,'open') AS question_type, is_required FROM teacher_post_questions WHERE is_active=1 ORDER BY display_order, id"
    ).fetchall()

    if request.method == "POST":
        errors = {}
        answers = {}

        for q in questions:
            key = f"q_{q['id']}"
            val = request.form.get(key)
            qtype = (q['question_type'] or 'open').lower()

            if q['is_required'] and (not val or not val.strip()):
                errors[q['id']] = "این مورد الزامی است."
            else:
                answers[q['id']] = val

        if errors:
            return render_template(
                'post_test_teacher.html',
                questions=questions,
                errors=errors,
                values=request.form
            )

        # ذخیره پاسخ‌ها
        for qid, val in answers.items():
            q = next((qq for qq in questions if qq['id']==qid), None)
            qtype = (q['question_type'] if q else 'open')
            if (qtype or 'open').lower() == 'likert':
                try:
                    aval = int(val) if val is not None and str(val).strip()!='' else None
                except ValueError:
                    aval = None
                atext = None
            else:
                aval = None
                atext = val

            cur.execute("""
                INSERT INTO teacher_post_answers (participant_id, question_id, answer_value, answer_text)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(participant_id, question_id) DO UPDATE SET
                    answer_value=excluded.answer_value,
                    answer_text=excluded.answer_text,
                    created_at=CURRENT_TIMESTAMP
            """, (pid, qid, aval, atext))
        conn.commit()
        conn.close()

        return redirect(url_for('result'))

    conn.close()
    return render_template('post_test_teacher.html', questions=questions, errors={}, values={})

@app.route('/post_test_manager', methods=['GET', 'POST'])
def post_test_manager():
    if 'participant_id' not in session or session.get('role') != 'manager':
        return redirect(url_for('index'))

    pid = session['participant_id']

    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # آیا قبلاً پر کرده؟
    row = cur.execute(
        "SELECT 1 FROM manager_post_answers WHERE participant_id=? LIMIT 1",
        (pid,)
    ).fetchone()

    if row:
        return render_template("post_test_completed.html", role="manager")

    # خواندن سوالات
    questions = cur.execute(
        "SELECT id, text, COALESCE(question_type,'open') AS question_type, is_required FROM manager_post_questions WHERE is_active=1 ORDER BY display_order, id"
    ).fetchall()

    if request.method == "POST":
        errors = {}
        answers = {}

        for q in questions:
            key = f"q_{q['id']}"
            val = request.form.get(key)
            qtype = (q['question_type'] or 'open').lower()

            if q['is_required'] and (not val or not val.strip()):
                errors[q['id']] = "این مورد الزامی است."
            else:
                answers[q['id']] = val

        if errors:
            return render_template(
                'post_test_manager.html',
                questions=questions,
                errors=errors,
                values=request.form
            )

        # ذخیره
        for qid, val in answers.items():
            q = next((qq for qq in questions if qq['id']==qid), None)
            qtype = (q['question_type'] if q else 'open')
            if (qtype or 'open').lower() == 'likert':
                try:
                    aval = int(val) if val is not None and str(val).strip()!='' else None
                except ValueError:
                    aval = None
                atext = None
            else:
                aval = None
                atext = val

            cur.execute("""
                INSERT INTO manager_post_answers (participant_id, question_id, answer_value, answer_text)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(participant_id, question_id) DO UPDATE SET
                    answer_value=excluded.answer_value,
                    answer_text=excluded.answer_text,
                    created_at=CURRENT_TIMESTAMP
            """, (pid, qid, aval, atext))
        conn.commit()
        conn.close()

        return redirect(url_for('result'))


    conn.close()
    return render_template('post_test_manager.html', questions=questions, errors={}, values={})


# ----------------------------- دانلود -----------------------------
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


# ----------------------------- اجرا -----------------------------
if __name__ == '__main__':
    if not os.path.exists(DATABASE):
        init_db()
    else:
        # ایندکس‌های لازم برای UPSERT voice/user_results را مطمئن شو
        with sqlite3.connect(DATABASE, timeout=30) as conn:
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_user_results_user ON user_results(user_id);"
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_voice_answers_user_q ON voice_answers(participant_id, question_id);"
            )
            conn.commit()
    app.run(debug=True, use_reloader=False, threaded=False)

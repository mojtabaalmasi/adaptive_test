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

# ----------------------------- پیکربندی پایه -----------------------------
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret')


DATABASE = 'questions.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    return conn

# آپلود ویس
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'voices')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
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

        """)
        # ایندکس‌های موردنیاز برای UPSERT
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_user_results_user ON user_results(user_id);")
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_voice_answers_user_q ON voice_answers(participant_id, question_id);")
        conn.commit()

# ----------------------------- لایهٔ سؤال/پارامترها -----------------------------
def get_question_by_id(question_id: int):
    conn = get_db_connection()
    row = conn.execute("SELECT id, text, option1, option2, option3, option4 FROM questions WHERE id = ?", (question_id,)).fetchone()
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
    info = (a ** 2) * ((p - c) ** 2) / ((1.0 - c) ** 2 + EPS) * (q / p)
    return float(info) if np.isfinite(info) and info > 0 else 0.0

def test_information(theta, item_params):
    return float(sum(item_information(theta, a, b, c) for (a, b, c) in item_params))

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

def estimate_theta_map(responses, item_params, theta0=0.0, prior_mean=0.0, prior_var=1.0, max_iter=50, tol=1e-4):
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

def theta_se(theta, item_params):
    I = test_information(theta, item_params)
    return float(1.0 / np.sqrt(max(I, EPS)))

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
    session.clear()
    session['answered_questions'] = []
    session['responses'] = []
    session['theta'] = 0.0
    return redirect(url_for('register'))

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

                conn.commit()

            session['participant_id'] = participant_id
            session['user_name'] = name
            session['role'] = role
            return redirect(url_for('manager_survey' if role == 'manager' else 'test'))

        except sqlite3.Error as e:
            print("DB Error:", e)
            return f"خطای پایگاه داده: {e}", 500
        except Exception as e:
            print("Error:", e)
            return f"خطا: {e}", 500
    return render_template('register.html')

@app.route('/manager_survey')
def manager_survey():
    if 'participant_id' not in session:
        return redirect(url_for('index'))
    if session.get('role') != 'manager':
        return redirect(url_for('test'))

    conn = get_db_connection()
    rows = conn.execute("""
        SELECT id, text
        FROM manager_questions
        WHERE is_active = 1
        ORDER BY display_order, id
    """).fetchall()
    conn.close()

    questions = [{'id': int(r['id']), 'text': r['text']} for r in rows]
    return render_template('manager_survey.html', questions=questions, user_name=session.get('user_name', ''))

@app.route('/api/voice_answer', methods=['POST'])
def api_voice_answer():
    if 'participant_id' not in session:
        return {'ok': False, 'error': 'unauthorized'}, 401

    participant_id = int(session['participant_id'])
    role = session.get('role', 'manager')
    question_id = request.form.get('question_id')
    duration_ms = request.form.get('duration_ms')
    f = request.files.get('audio')

    if not question_id or not f:
        return {'ok': False, 'error': 'missing fields'}, 400

    mime = f.mimetype or 'audio/webm'
    if mime not in ALLOWED_MIME:
        return {'ok': False, 'error': f'unsupported mime {mime}'}, 400

    ext = MIME_EXT.get(mime, 'webm')
    filename = secure_filename(f"{participant_id}_{question_id}_{uuid.uuid4().hex}.{ext}")
    save_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(participant_id))
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    f.save(path)

    try:
        size_bytes = os.path.getsize(path)
    except Exception:
        size_bytes = None

    rel_path = os.path.relpath(path, start=os.path.join(app.root_path, 'static')).replace('\\', '/')

    with sqlite3.connect(DATABASE, timeout=30) as conn:
        conn.execute("PRAGMA busy_timeout=30000;")
        cur = conn.cursor()
        # UPSERT (نیازمند UNIQUE(participant_id, question_id))
        try:
            cur.execute("""
                INSERT INTO voice_answers (participant_id, role, question_id, file_path, mime_type, duration_ms, size_bytes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(participant_id, question_id)
                DO UPDATE SET file_path=excluded.file_path,
                              mime_type=excluded.mime_type,
                              duration_ms=excluded.duration_ms,
                              size_bytes=excluded.size_bytes,
                              created_at=CURRENT_TIMESTAMP
            """, (participant_id, role, int(question_id), rel_path, mime, int(duration_ms) if duration_ms else None, size_bytes))
        except sqlite3.OperationalError:
            # اگر UNIQUE وجود ندارد: حذف قبلی و درج جدید
            cur.execute("DELETE FROM voice_answers WHERE participant_id=? AND question_id=?", (participant_id, int(question_id)))
            cur.execute("""
                INSERT INTO voice_answers (participant_id, role, question_id, file_path, mime_type, duration_ms, size_bytes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (participant_id, role, int(question_id), rel_path, mime, int(duration_ms) if duration_ms else None, size_bytes))
        conn.commit()

    url = url_for('static', filename=rel_path)
    return {'ok': True, 'url': url}

# ----------------------------- آزمون تطبیقی -----------------------------
@app.route('/test', methods=['GET', 'POST'])
def test():
    # دسترسی
    if 'participant_id' not in session:
        return redirect(url_for('index'))

    # آماده‌سازی سشن
    session.setdefault('answered_questions', [])
    session.setdefault('responses', [])
    session.setdefault('theta', 0.0)
    session.setdefault('stable_streak', 0)

    answered = list(map(int, session['answered_questions']))  # اندیس‌های 0-مبنا
    responses = list(map(int, session['responses']))          # 0/1
    theta = float(session['theta'])
    streak = int(session['stable_streak'])

    # بانک سؤال
    rows = get_all_item_params()                    # [(id, a, b, c), ...]
    if not rows:
        flash('بانک سؤال خالی است.', 'error')
        return redirect(url_for('index'))
    question_ids = [r[0] for r in rows]             # شناسه‌های واقعی DB
    all_item_params = [tuple(r[1:]) for r in rows]  # [(a,b,c), ...]

    total_questions = len(all_item_params)

    # تنظیمات CAT
    MIN_QUESTIONS = 8          # حداقل آیتم قبل از شروع ارزیابی توقف
    HARD_MAX     = 22          # سقف ایمنی
    SE_TARGET    = 0.30        # هدف دقت
    DELTA_TARGET = 0.03        # آستانه پایداری Δθ
    STREAK_NEED  = 2           # چند بار پیاپی Δθ کوچک
    CUT_SCORE    = 0.0         # مرز (مثلاً قبولی = 0)
    Z_CI         = 1.96        # 95% CI

    # ------------------- GET: شروع آزمون (اگر هنوز شروع نشده) -------------------
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

    # ------------------- POST: ثبت پاسخ -------------------
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

        # سؤال فعلی (شناسهٔ واقعی)
        current_idx = answered[-1]
        current_qid = question_ids[current_idx]

        # درست/نادرست
        co = get_correct_answer(current_qid)
        try:
            co_int = int(co)
        except (TypeError, ValueError):
            co_int = None
        is_correct = 1 if (co_int is not None and sel == co_int) else 0
        responses.append(is_correct)

        # برآورد θ (ابتدا MAP پایدارتر است)
        answered_params = [all_item_params[i] for i in answered]
        old_theta = theta
        if len(responses) < 3:
            theta = estimate_theta_map(responses, answered_params, prior_mean=0.0, prior_var=1.0)
        else:
            theta = estimate_theta_mle(responses, answered_params)
        theta_change = abs(theta - old_theta)
        se_now = theta_se(theta, answered_params)

        # ذخیره پاسخ در DB (سازگار با شِمای قدیمی که response NOT NULL دارد)
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
            db.commit()

        # به‌روزرسانی پایداری Δθ
        if theta_change < DELTA_TARGET:
            streak += 1
        else:
            streak = 0
        session['stable_streak'] = int(streak)

        # معیارهای توقف
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
            # ذخیره نتیجهٔ نهایی
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
            session['stop_reason'] = stop_reason
            session.modified = True
            return redirect(url_for('result'))

        # انتخاب سؤال بعدی
        next_idx = select_next_question(theta, all_item_params, answered)
        if next_idx is None:
            # بانک تمام شد
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

        # ادامه آزمون
        answered.append(next_idx)
        session['answered_questions'] = answered
        session['responses'] = responses
        session['theta'] = float(theta)
        session.modified = True

        next_qid = question_ids[next_idx]
        question = get_question_by_id(next_qid)
        progress = int(len(answered) / max(total_questions, 1) * 100)
        return render_template('test.html', question=question, progress=progress)

    # ------------------- نمایش سؤال جاری (در ادامهٔ GET) -------------------
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

    # سؤالات از جدول strategies (گروه‌بندی بر اساس category)
    with sqlite3.connect(DATABASE, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000;")
        cur = conn.cursor()
        rows = cur.execute("""
            SELECT id, strategy AS text, COALESCE(category, '') AS category
            FROM strategies
            ORDER BY category, id
        """).fetchall()

    # ساخت گروه‌ها برای نمایش تمیز
    groups = []
    last_cat = None
    for r in rows:
        cat = r['category']
        if cat != last_cat:
            groups.append({'category': cat, 'items': []})
            last_cat = cat
        groups[-1]['items'].append({'id': int(r['id']), 'text': r['text']})

    # اگر پرسشنامه خالی بود
    if not rows:
        flash('هیچ موردی در پرسشنامهٔ راهبردها تعریف نشده است.', 'error')
        return redirect(url_for('result'))

    if request.method == 'POST':
        errors = {}
        payload = {}
        # همهٔ آیتم‌ها را «الزامی» فرض کرده‌ایم؛ اگر نمی‌خواهی، این بخش را سفارشی کن
        for g in groups:
            for item in g['items']:
                key = f"s_{item['id']}"
                val = request.form.get(key)
                if not val:
                    errors[item['id']] = "الزامی"
                    continue
                try:
                    choice = int(val)
                except ValueError:
                    errors[item['id']] = "نامعتبر"
                    continue
                if choice < 1 or choice > 5:
                    errors[item['id']] = "بازه ۱ تا ۵"
                    continue
                payload[item['id']] = choice

        if errors:
            # نمایش دوباره با خطا و مقادیر انتخاب‌شده
            return render_template('strategies_survey.html',
                                   groups=groups, errors=errors, values=request.form)

        # ذخیرهٔ پاسخ‌ها (UPSERT روی (participant_id, strategy_id))
        with sqlite3.connect(DATABASE, timeout=30) as conn:
            conn.execute("PRAGMA busy_timeout=30000;")
            cur = conn.cursor()
            try:
                for sid, choice in payload.items():
                    cur.execute("""
                        INSERT INTO strategy_answers (participant_id, strategy_id, choice)
                        VALUES (?, ?, ?)
                        ON CONFLICT(participant_id, strategy_id)
                        DO UPDATE SET choice=excluded.choice,
                                      updated_at=CURRENT_TIMESTAMP
                    """, (participant_id, sid, choice))
            except sqlite3.OperationalError:
                # اگر قید یکتا موجود نباشد، fallback: حذف-و-درج
                for sid, choice in payload.items():
                    cur.execute("DELETE FROM strategy_answers WHERE participant_id=? AND strategy_id=?",
                                (participant_id, sid))
                    cur.execute("""
                        INSERT INTO strategy_answers (participant_id, strategy_id, choice)
                        VALUES (?, ?, ?)
                    """, (participant_id, sid, choice))
            conn.commit()

        # بعد از ذخیرهٔ پاسخ‌ها
        session['post_test_saved'] = True
        return redirect(url_for('thank_you'))


    rt_ms = request.form.get('rt_ms')
    try: rt_ms = int(rt_ms)
    except: rt_ms = None

    info_here = item_information(theta, *all_item_params[current_idx])  # قبل از آپدیت θ

    # درج در answers_meta
    with sqlite3.connect(DATABASE, timeout=30) as conn:
        conn.execute("PRAGMA busy_timeout=30000;")
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO answers_meta
            (session_id, user_id, question_id, order_idx,
            selected_option, is_correct, theta_before, theta_after,
            se_after, info_at_theta, rt_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (session['cat_session_id'], session['participant_id'], current_qid,
            len(responses), sel, is_correct, float(old_theta), float(theta),
            float(se_now), float(info_here), rt_ms))
        conn.commit()

    # GET
    return render_template('strategies_survey.html', groups=groups, errors={}, values={})

# ----------------------------- قدردانی -----------------------------

@app.route('/thank_you')
def thank_you():
    if 'participant_id' not in session:
        # اگر کسی مستقیم آمد
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
    rows = get_all_item_params()                       # [(id, a, b, c), ...]
    all_item_params = [tuple(r[1:]) for r in rows]     # [(a,b,c), ...]
    answered_params = [all_item_params[i] for i in answered] if answered else []

    # نمودارها
    icc_path = plot_icc(answered_params, save_path=f'static/icc_{uuid.uuid4().hex}.png') if answered_params else None
    info_path = plot_item_information(answered_params, save_path=f'static/info_{uuid.uuid4().hex}.png') if answered_params else None

    # آمار پایه
    n_total   = len(responses)
    n_correct = int(sum(1 for r in responses if r == 1))
    n_wrong   = int(n_total - n_correct)
    accuracy  = float(round((n_correct / n_total) * 100.0, 1)) if n_total > 0 else 0.0

    # SE و بازه‌های اطمینان
    if answered_params:
        se = theta_se(theta, answered_params)
        ci68 = (max(-4.0, theta - se),  min(4.0, theta + se))
        ci95 = (max(-4.0, theta - 1.96*se), min(4.0, theta + 1.96*se))
    else:
        se, ci68, ci95 = None, None, None

    # تفسیر لایه‌بندی‌شده
    def ability_band(t):
        if t < -2.0:      return "خیلی پایین"
        elif t < -1.0:    return "پایین"
        elif t <= 1.0:    return "متوسط"
        elif t <= 2.0:    return "بالا"
        else:             return "خیلی بالا"

    band = ability_band(theta)

    # تفسیر متنی کمی مفصل‌تر
    if band == "خیلی پایین":
        interpretation = "نتیجه نشان می‌دهد سطح توانایی شما در این حیطه بسیار پایین است. پیشنهاد می‌شود با آیتم‌های بسیار ساده‌تر و مرور مفاهیم پایه شروع کنید و سپس تدریجاً دشواری را بالا ببرید."
    elif band == "پایین":
        interpretation = "سطح توانایی شما پایین‌تر از میانگین است. تمرین هدفمند روی موضوعاتی که اشتباه بیشتری داشتید، می‌تواند سریع‌ترین بهبود را ایجاد کند."
    elif band == "متوسط":
        interpretation = "توانایی شما نزدیک به میانگین شرکت‌کنندگان است. با ادامه تمرین و قرار گرفتن در معرض آیتم‌های کمی دشوارتر، می‌توانید θ را افزایش دهید."
    elif band == "بالا":
        interpretation = "عملکرد شما بالاتر از میانگین است. آیتم‌های چالش‌برانگیزتر می‌توانند تمایز دقیق‌تری از توانایی‌تان ارائه دهند."
    else:  # خیلی بالا
        interpretation = "توانایی شما بسیار بالا برآورد شده است. برای تفکیک بیشتر، آیتم‌های بسیار دشوار و سنجه‌های پیشرفته‌تر توصیه می‌شود."

    user_name = session.get('user_name', 'کاربر ناشناس')

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
        interpretation=interpretation
    )


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
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_user_results_user ON user_results(user_id);")
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_voice_answers_user_q ON voice_answers(participant_id, question_id);")
            conn.commit()
    app.run(debug=True, use_reloader=False, threaded=False)

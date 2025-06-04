import sqlite3
import json
from datetime import datetime

DB_PATH = 'questions.db'  # مسیر دیتابیس شما

def create_results_table():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            native_language TEXT,
            major TEXT,
            age INTEGER,
            persian_familiarity TEXT,
            theta REAL,
            responses TEXT,
            test_date TEXT
        )
    ''')
    conn.commit()
    conn.close()
    print("جدول user_results ساخته شد (اگر قبلاً نبود).")

def insert_user_result(full_name, native_language, major, age, persian_familiarity, theta, responses):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # تبدیل پاسخ‌ها به رشته JSON برای ذخیره راحت
    responses_json = json.dumps(responses, ensure_ascii=False)

    cursor.execute('''
        INSERT INTO user_results 
        (full_name, native_language, major, age, persian_familiarity, theta, responses, test_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (full_name, native_language, major, age, persian_familiarity, theta, responses_json, datetime.now().isoformat()))

    conn.commit()
    conn.close()
    print("نتایج کاربر در دیتابیس ذخیره شد.")

if __name__ == "__main__":
    create_results_table()
    # برای تست:
    # insert_user_result("مریم رضایی", "فارسی", "مهندسی کامپیوتر", 25, "زیاد", 1.23, [1,0,1,1])

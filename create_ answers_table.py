import sqlite3

# مسیر فایل پایگاه داده
db_path = "questions.db"

# اتصال به پایگاه داده
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# ساخت جدول اگر وجود نداشته باشد
create_table_query = '''
CREATE TABLE IF NOT EXISTS answers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    participant_id INTEGER,
    question_id INTEGER,
    selected_option INTEGER,
    is_correct BOOLEAN,
    FOREIGN KEY (participant_id) REFERENCES participants(id),
    FOREIGN KEY (question_id) REFERENCES questions(id)
);
'''

# اجرای دستور
cursor.execute(create_table_query)
conn.commit()

print("✅ جدول 'answers' با موفقیت ایجاد شد.")

# بستن اتصال
conn.close()

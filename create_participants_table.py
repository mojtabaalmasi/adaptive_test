import sqlite3

conn = sqlite3.connect('questions.db')
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS participants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    language TEXT,
    major TEXT,
    age INTEGER,
    farsi_level TEXT,
    farsi_skills TEXT,
    farsi_courses TEXT,
    learning_place TEXT
)
""")

conn.commit()
conn.close()

print("✅ جدول 'participants' با موفقیت ساخته شد.")

import sqlite3
import sys

def main():
    try:
        conn = sqlite3.connect('Data/storage.sqlite')
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in cursor.fetchall()]
        print("TABLES:", tables)
        for t in tables:
            cursor.execute(f"PRAGMA table_info({t});")
            cols = [c[1] for c in cursor.fetchall()]
            print(f"Table {t} cols: {cols}")
            
            cursor.execute(f"SELECT * FROM {t} LIMIT 1")
            row = cursor.fetchone()
            print(f"Sample row {t}:", row)
        conn.close()
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()

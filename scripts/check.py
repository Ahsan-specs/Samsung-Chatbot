import sqlite3
import sys

def main():
    print("Opening database...")
    sys.stdout.flush()
    try:
        conn = sqlite3.connect('storage.sqlite')
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in cursor.fetchall()]
        print("TABLES:", tables)
        for t in tables:
            cursor.execute(f"PRAGMA table_info({t});")
            cols = [c[1] for c in cursor.fetchall()]
            print(f"[{t}] Columns: {cols}")
            
            cursor.execute(f"SELECT * FROM {t} LIMIT 1")
            row = cursor.fetchone()
            print(f"[{t}] Sample:", repr(row)[:200])
        conn.close()
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()

"""Temporary script to search VS Code Copilot chat session history."""
import sqlite3
import json

db_path = r'C:\Users\c3205\AppData\Roaming\Code\User\workspaceStorage\f2e74b59c4a8a422ff64a4d480554ad1\state.vscdb'
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# List all tables
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]
print("Tables:", tables)

# Check the ItemTable for chat-related keys
for table in tables:
    cur.execute(f"SELECT count(*) FROM [{table}]")
    count = cur.fetchone()[0]
    print(f"\n{table}: {count} rows")
    
    # Get column names
    cur.execute(f"PRAGMA table_info([{table}])")
    cols = [(r[1], r[2]) for r in cur.fetchall()]
    print(f"  Columns: {cols}")
    
    # Look for chat-related keys
    if any('key' in c[0].lower() for c in cols):
        key_col = [c[0] for c in cols if 'key' in c[0].lower()][0]
        cur.execute(f"SELECT [{key_col}] FROM [{table}] WHERE [{key_col}] LIKE '%chat%' OR [{key_col}] LIKE '%copilot%' OR [{key_col}] LIKE '%session%' LIMIT 20")
        matches = cur.fetchall()
        if matches:
            print(f"  Chat-related keys:")
            for m in matches:
                print(f"    {m[0][:120]}")

conn.close()

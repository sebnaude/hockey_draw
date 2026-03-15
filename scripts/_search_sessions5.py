"""Deep search for chat session content - check all possible storage locations."""
import sqlite3
import json
import os
import glob

# Check all possible storage locations for chat history

# 1. Check chatEditingSessions folders
ws_dir = r'C:\Users\c3205\AppData\Roaming\Code\User\workspaceStorage\f2e74b59c4a8a422ff64a4d480554ad1'
edit_sessions = os.path.join(ws_dir, 'chatEditingSessions')
if os.path.exists(edit_sessions):
    for d in os.listdir(edit_sessions):
        state_file = os.path.join(edit_sessions, d, 'state.json')
        if os.path.exists(state_file):
            with open(state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"chatEditingSessions/{d}: keys={list(data.keys())[:10]}")

# 2. Check global storage
global_dir = r'C:\Users\c3205\AppData\Roaming\Code\User\globalStorage\github.copilot-chat'
for f in os.listdir(global_dir):
    full = os.path.join(global_dir, f)
    if os.path.isfile(full) and f.endswith('.json') and 'session' in f.lower():
        print(f"\nGlobal: {f} ({os.path.getsize(full)} bytes)")

# 3. Check the interactive-session structure more carefully  
db_path = os.path.join(ws_dir, 'state.vscdb')
conn = sqlite3.connect(db_path)
cur = conn.cursor()

cur.execute("SELECT value FROM ItemTable WHERE key='memento/interactive-session'")
row = cur.fetchone()
raw = row[0] if isinstance(row[0], str) else row[0].decode('utf-8')
data = json.loads(raw)

sessions = data.get('history', {}).get('copilot', [])
print(f"\n=== INTERACTIVE SESSION STRUCTURE ===")
print(f"Total sessions: {len(sessions)}")

for i, session in enumerate(sessions[:3]):
    print(f"\nSession {i}: keys = {list(session.keys())}")
    sid = session.get('sessionId', '?')
    requests = session.get('requests', [])
    print(f"  sessionId: {sid}")
    print(f"  requests: {len(requests)}")
    if requests:
        req = requests[0]
        print(f"  First request keys: {list(req.keys())}")
        msg = req.get('message', {})
        print(f"  Message type: {type(msg).__name__}")
        if isinstance(msg, dict):
            print(f"  Message keys: {list(msg.keys())}")
            text = msg.get('text', '')
            print(f"  Message text: {text[:200]}")
        resp = req.get('response', {})
        if isinstance(resp, dict):
            print(f"  Response keys: {list(resp.keys())}")

# 4. Check if there are separate chat session files stored elsewhere
session_dirs = glob.glob(os.path.join(ws_dir, 'chat*'))
print(f"\n=== Chat-related dirs in workspace storage ===")
for d in session_dirs:
    print(f"  {os.path.basename(d)}")
    if os.path.isdir(d):
        for item in os.listdir(d)[:5]:
            full = os.path.join(d, item)
            print(f"    {item} ({'dir' if os.path.isdir(full) else f'{os.path.getsize(full)} bytes'})")

# 5. Check the state.vscdb for ALL keys and their sizes
print(f"\n=== ALL keys by size (descending) ===")
cur.execute("SELECT key, length(value) as sz FROM ItemTable ORDER BY sz DESC")
for k, sz in cur.fetchall()[:30]:
    print(f"  {sz:>10} bytes | {k}")

conn.close()

"""Search all chat session titles and content for 'club' + 'close/together/proximity'."""
import sqlite3
import json
from datetime import datetime

db_path = r'C:\Users\c3205\AppData\Roaming\Code\User\workspaceStorage\f2e74b59c4a8a422ff64a4d480554ad1\state.vscdb'
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# Get session index
cur.execute("SELECT value FROM ItemTable WHERE key='chat.ChatSessionStore.index'")
row = cur.fetchone()
raw = row[0] if isinstance(row[0], str) else row[0].decode('utf-8')
index = json.loads(raw)

print("=" * 80)
print(f"ALL SESSIONS ({len(index['entries'])} total)")
print("=" * 80)

entries = sorted(index['entries'].values(), key=lambda e: e.get('timing', {}).get('created', 0))

for i, entry in enumerate(entries):
    created = entry.get('timing', {}).get('created', 0)
    if created > 0:
        dt = datetime.fromtimestamp(created / 1000)
        date_str = dt.strftime('%Y-%m-%d %H:%M')
    else:
        date_str = 'unknown'
    title = entry.get('title', 'No title')
    sid = entry['sessionId']
    print(f"{i+1:3}. [{date_str}] {title}")
    print(f"     ID: {sid}")

# Now search actual session content for keywords
print("\n" + "=" * 80)
print("SEARCHING SESSION CONTENT for 'club' + 'close/together/proximity/density/grouped'")
print("=" * 80)

# Sessions are stored in the interactive-session history
cur.execute("SELECT value FROM ItemTable WHERE key='memento/interactive-session'")
row = cur.fetchone()
if row:
    raw = row[0] if isinstance(row[0], str) else row[0].decode('utf-8')
    history = json.loads(raw)
    
    if 'history' in history and 'copilot' in history['history']:
        copilot_sessions = history['history']['copilot']
        if isinstance(copilot_sessions, list):
            print(f"Found {len(copilot_sessions)} sessions in history")
        elif isinstance(copilot_sessions, dict):
            print(f"Found dict with keys: {list(copilot_sessions.keys())[:10]}")

# Also check each session key individually
print("\n--- Checking all ItemTable keys for session data ---")
cur.execute("SELECT key FROM ItemTable WHERE key LIKE '%session%' OR key LIKE '%chat%'")
keys = [r[0] for r in cur.fetchall()]
for k in keys:
    cur.execute("SELECT length(value) FROM ItemTable WHERE key=?", (k,))
    size = cur.fetchone()[0]
    print(f"  {k}: {size} bytes")

# Search for session content stored per-session-id
for entry in entries:
    sid = entry['sessionId']
    # Try various key patterns
    for prefix in ['chat.session.', 'chatSession.', '']:
        key = f"{prefix}{sid}"
        cur.execute("SELECT value FROM ItemTable WHERE key=?", (key,))
        row = cur.fetchone()
        if row:
            print(f"\nFound data for session {sid} at key '{key}'")

# Check GitHub.copilot-chat key
cur.execute("SELECT value FROM ItemTable WHERE key='GitHub.copilot-chat'")
row = cur.fetchone()
if row:
    raw = row[0] if isinstance(row[0], str) else row[0].decode('utf-8')
    data = json.loads(raw)
    print(f"\nGitHub.copilot-chat: type={type(data).__name__}")
    if isinstance(data, dict):
        for k in list(data.keys())[:20]:
            v = data[k]
            print(f"  {k}: {type(v).__name__} ({str(v)[:80] if not isinstance(v, (dict, list)) else '...'})")

conn.close()

"""Search VS Code Copilot chat session history for keyword matches."""
import sqlite3
import json
import sys

db_path = r'C:\Users\c3205\AppData\Roaming\Code\User\workspaceStorage\f2e74b59c4a8a422ff64a4d480554ad1\state.vscdb'
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# Get the chat session index
cur.execute("SELECT value FROM ItemTable WHERE key='chat.ChatSessionStore.index'")
row = cur.fetchone()
if row:
    data = json.loads(row[0]) if isinstance(row[0], str) else json.loads(row[0].decode('utf-8'))
    print(f"Chat Session Index: {len(data) if isinstance(data, list) else 'dict'}")
    print(json.dumps(data, indent=2, default=str)[:3000])
    print("...")
else:
    print("No chat session index found")

print("\n" + "="*60)

# Check interactive session memento
for key in ['memento/interactive-session', 'memento/interactive-session-view-copilot']:
    cur.execute("SELECT value FROM ItemTable WHERE key=?", (key,))
    row = cur.fetchone()
    if row:
        raw = row[0] if isinstance(row[0], str) else row[0].decode('utf-8')
        data = json.loads(raw)
        print(f"\n{key}: type={type(data).__name__}")
        if isinstance(data, dict):
            print(f"  Keys: {list(data.keys())[:20]}")
            # Look for sessions list
            for k, v in data.items():
                if isinstance(v, list):
                    print(f"  {k}: list of {len(v)} items")
                elif isinstance(v, dict):
                    print(f"  {k}: dict with keys {list(v.keys())[:10]}")
                else:
                    print(f"  {k}: {str(v)[:100]}")

# Also check agentSessions
cur.execute("SELECT value FROM ItemTable WHERE key='agentSessions.state.cache'")
row = cur.fetchone()
if row:
    raw = row[0] if isinstance(row[0], str) else row[0].decode('utf-8')
    data = json.loads(raw)
    print(f"\nagentSessions.state.cache: type={type(data).__name__}")
    if isinstance(data, dict):
        print(f"  Keys: {list(data.keys())[:20]}")
        for k in list(data.keys())[:5]:
            v = data[k]
            if isinstance(v, dict):
                print(f"  {k}: {list(v.keys())[:10]}")

conn.close()

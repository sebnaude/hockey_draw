"""Search chat session MESSAGE CONTENT for keywords about clubs playing close together."""
import sqlite3
import json
import re
from datetime import datetime

db_path = r'C:\Users\c3205\AppData\Roaming\Code\User\workspaceStorage\f2e74b59c4a8a422ff64a4d480554ad1\state.vscdb'
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# Get the interactive session data which contains actual messages
cur.execute("SELECT value FROM ItemTable WHERE key='memento/interactive-session'")
row = cur.fetchone()
raw = row[0] if isinstance(row[0], str) else row[0].decode('utf-8')
data = json.loads(raw)

# Get session index for titles/dates
cur.execute("SELECT value FROM ItemTable WHERE key='chat.ChatSessionStore.index'")
row2 = cur.fetchone()
raw2 = row2[0] if isinstance(row2[0], str) else row2[0].decode('utf-8')
index = json.loads(raw2)

sessions = data.get('history', {}).get('copilot', [])

# Search keywords
keywords = re.compile(r'(club.*close|close.*together|club.*proximity|club.*density|clubs.*play.*near|clubs.*grouped|play.*close.*together|force.*club|new.*constraint.*club|constraint.*force.*club|club.*timeslot.*gap|club.*spread|club.*compact|club.*sequential|sequential.*club|grouped.*sequential|minimise.*gap.*club|minimize.*gap.*club)', re.IGNORECASE)

print(f"Searching {len(sessions)} sessions for keyword matches...")
print("=" * 80)

for session in sessions:
    session_id = session.get('sessionId', 'unknown')
    requests = session.get('requests', [])
    
    # Get title from index
    entry = index.get('entries', {}).get(session_id, {})
    title = entry.get('title', 'Unknown')
    created = entry.get('timing', {}).get('created', 0)
    date_str = datetime.fromtimestamp(created/1000).strftime('%Y-%m-%d') if created > 0 else '?'
    
    matches_found = []
    for req in requests:
        # Check user message
        msg = req.get('message', {})
        if isinstance(msg, dict):
            text = msg.get('text', '')
        elif isinstance(msg, str):
            text = msg
        else:
            text = str(msg)
        
        if keywords.search(text):
            # Extract a snippet around the match
            m = keywords.search(text)
            start = max(0, m.start() - 80)
            end = min(len(text), m.end() + 80)
            snippet = text[start:end].replace('\n', ' ')
            matches_found.append(('USER', snippet))
        
        # Check response
        response = req.get('response', {})
        if isinstance(response, dict):
            resp_parts = response.get('value', [])
            if isinstance(resp_parts, list):
                for part in resp_parts:
                    if isinstance(part, dict):
                        resp_text = part.get('value', '')
                    elif isinstance(part, str):
                        resp_text = part
                    else:
                        continue
                    if keywords.search(str(resp_text)):
                        m = keywords.search(str(resp_text))
                        start = max(0, m.start() - 80)
                        end = min(len(str(resp_text)), m.end() + 80)
                        snippet = str(resp_text)[start:end].replace('\n', ' ')
                        matches_found.append(('RESPONSE', snippet))
    
    if matches_found:
        print(f"\n[{date_str}] {title}")
        print(f"  Session ID: {session_id}")
        for role, snippet in matches_found[:5]:
            print(f"  {role}: ...{snippet}...")

# Also do a broader search - just dump message text and search
print("\n\n" + "=" * 80)
print("BROADER SEARCH: all user messages containing 'constraint' + 'club'")
print("=" * 80)

for session in sessions:
    session_id = session.get('sessionId', 'unknown')
    entry = index.get('entries', {}).get(session_id, {})
    title = entry.get('title', 'Unknown')
    created = entry.get('timing', {}).get('created', 0)
    date_str = datetime.fromtimestamp(created/1000).strftime('%Y-%m-%d') if created > 0 else '?'
    
    requests = session.get('requests', [])
    for req in requests:
        msg = req.get('message', {})
        text = msg.get('text', '') if isinstance(msg, dict) else str(msg)
        
        if 'constraint' in text.lower() and 'club' in text.lower():
            # Show first 300 chars
            print(f"\n[{date_str}] {title}")
            print(f"  Session ID: {session_id}")
            print(f"  Message: {text[:400].replace(chr(10), ' ')}")
            print("  ---")

conn.close()

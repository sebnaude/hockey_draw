"""Search all chat session JSONL files for keywords about clubs playing close together."""
import os
import json
import re
from datetime import datetime

sessions_dir = r'C:\Users\c3205\AppData\Roaming\Code\User\workspaceStorage\f2e74b59c4a8a422ff64a4d480554ad1\chatSessions'

# Get session index for titles
import sqlite3
db_path = r'C:\Users\c3205\AppData\Roaming\Code\User\workspaceStorage\f2e74b59c4a8a422ff64a4d480554ad1\state.vscdb'
conn = sqlite3.connect(db_path)
cur = conn.cursor()
cur.execute("SELECT value FROM ItemTable WHERE key='chat.ChatSessionStore.index'")
row = cur.fetchone()
raw = row[0] if isinstance(row[0], str) else row[0].decode('utf-8')
index_data = json.loads(raw)
entries = index_data.get('entries', {})
conn.close()

# Keywords to search for
patterns = [
    re.compile(r'club.*close.*together', re.IGNORECASE),
    re.compile(r'close.*together.*club', re.IGNORECASE),
    re.compile(r'force.*club.*play.*close', re.IGNORECASE),
    re.compile(r'club.*games.*close', re.IGNORECASE),
    re.compile(r'play.*close.*together', re.IGNORECASE),
    re.compile(r'clubs.*grouped', re.IGNORECASE),
    re.compile(r'club.*sequential', re.IGNORECASE),
    re.compile(r'sequential.*club', re.IGNORECASE),
    re.compile(r'new.*constraint.*club', re.IGNORECASE),
    re.compile(r'design.*constraint', re.IGNORECASE),
    re.compile(r'constraint.*design', re.IGNORECASE),
    re.compile(r'club.*proximity', re.IGNORECASE),
    re.compile(r'club.*density', re.IGNORECASE),
    re.compile(r'club.*compact', re.IGNORECASE),
    re.compile(r'minimis.*gap.*club', re.IGNORECASE),
    re.compile(r'club.*gap.*minimis', re.IGNORECASE),
    re.compile(r'club.*timeslot.*spread', re.IGNORECASE),
    re.compile(r'club.*spread', re.IGNORECASE),
    re.compile(r'club.*back.to.back', re.IGNORECASE),
    re.compile(r'club.*consecutive', re.IGNORECASE),
    re.compile(r'all.*club.*team.*play.*near', re.IGNORECASE),
    re.compile(r'club.*team.*same.*time', re.IGNORECASE),
]

for fname in sorted(os.listdir(sessions_dir)):
    if not fname.endswith(('.jsonl', '.json')):
        continue
    
    sid = fname.replace('.jsonl', '').replace('.json', '')
    entry = entries.get(sid, {})
    title = entry.get('title', 'Unknown')
    created = entry.get('timing', {}).get('created', 0)
    date_str = datetime.fromtimestamp(created/1000).strftime('%Y-%m-%d %H:%M') if created > 0 else '?'
    
    fpath = os.path.join(sessions_dir, fname)
    
    try:
        with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception as e:
        continue
    
    found_matches = []
    for pattern in patterns:
        for m in pattern.finditer(content):
            start = max(0, m.start() - 100)
            end = min(len(content), m.end() + 100)
            snippet = content[start:end].replace('\n', ' ').replace('\r', '')
            found_matches.append((pattern.pattern, snippet))
    
    if found_matches:
        print(f"\n{'='*80}")
        print(f"[{date_str}] {title}")
        print(f"File: {fname} ({os.path.getsize(fpath)} bytes)")
        print(f"Session ID: {sid}")
        seen = set()
        for pat, snippet in found_matches[:10]:
            key = snippet[:50]
            if key not in seen:
                seen.add(key)
                print(f"  Pattern: {pat}")
                print(f"  ...{snippet}...")
                print()

"""Search JSONL session files for club-playing-close-together keywords."""
import os
import json
import re
import sys

sessions_dir = r'C:\Users\c3205\AppData\Roaming\Code\User\workspaceStorage\f2e74b59c4a8a422ff64a4d480554ad1\chatSessions'

# Session index for titles
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

from datetime import datetime

keywords = [
    'close together',
    'club.*sequential',
    'sequential.*club',
    'grouped.*sequential',
    'club.*density',
    'club.*compact',
    'club.*proximity',
    'design.*new.*constraint',
    'new.*constraint',
    'club.*spread',
    'club.*gap',
    'play.*close',
    'back.to.back',
    'club day',
    'colts.*request',
    'colts.*sequential',
    'club.*team.*near',
    'force.*club',
    'constraint.*force',
]

combined = re.compile('|'.join(keywords), re.IGNORECASE)

output_file = r'C:\Users\c3205\Documents\Code\python\draw\scripts\_search_results.txt'
results = []

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
    except:
        continue

    matches = list(combined.finditer(content))
    if matches:
        snippets = []
        seen = set()
        for m in matches:
            start = max(0, m.start() - 120)
            end = min(len(content), m.end() + 120)
            snippet = content[start:end].replace('\n', ' ').replace('\r', '').strip()
            short = snippet[:60]
            if short not in seen:
                seen.add(short)
                snippets.append(snippet)

        results.append((date_str, title, sid, fname, len(matches), snippets))

# Output results to file
with open(output_file, 'w', encoding='utf-8') as out:
    out.write(f"Found {len(results)} sessions with matches\n\n")
    for date_str, title, sid, fname, count, snippets in results:
        out.write(f"[{date_str}] {title}\n")
        out.write(f"  File: {fname} | Matches: {count}\n")
        for s in snippets[:5]:
            out.write(f"  >>> {s[:250]}\n")
        out.write("\n")
print(f"Results written to {output_file}")

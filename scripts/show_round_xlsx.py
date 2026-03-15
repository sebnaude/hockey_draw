#!/usr/bin/env python3
"""Show games for a specific round from an xlsx draw file."""
import sys
import pandas as pd

def show_round(xlsx_path: str, round_no: int):
    df = pd.read_excel(xlsx_path)
    r = df[df['ROUND'] == round_no].sort_values(['DATE', 'TIME', 'LOCATION', 'FIELD'])
    
    print(f"=== ROUND {round_no} ({len(r)} games) ===\n")
    
    current_venue = None
    for _, row in r.iterrows():
        venue = str(row['LOCATION'])[:30]
        if venue != current_venue:
            print(f"\n{venue}")
            print("-" * 70)
            current_venue = venue
        
        field = str(row['FIELD'])[:6]
        time = str(row['TIME'])[:5]
        grade = str(row['GRADE'])[:4]
        home = str(row['TEAM 1'])[:22]
        away = str(row['TEAM 2'])[:22]
        print(f"  {time:<5} | {field:<6} | {grade:<4} | {home:<22} vs {away}")

if __name__ == "__main__":
    xlsx_path = sys.argv[1]
    round_no = int(sys.argv[2])
    show_round(xlsx_path, round_no)

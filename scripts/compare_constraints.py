#!/usr/bin/env python
"""Compare constraint counts between original and AI implementations.

Uses model.Proto().constraints length (same method as main_staged.py)
to accurately count OR-Tools constraints added.
"""

import os
import sys
# Add parent directory to path for imports from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_staged import load_data, STAGES, STAGES_AI
from ortools.sat.python import cp_model
from utils import generate_X

def count_constraints_applied(model, constraint_classes, X, data):
    """Count constraints using model.Proto().constraints length (like main_staged.py)."""
    total = 0
    results = []
    
    for c in constraint_classes:
        prior_count = len(model.Proto().constraints)
        instance = c()
        instance.apply(model, X, data)
        current_count = len(model.Proto().constraints)
        added = current_count - prior_count
        results.append((c.__name__, added))
        total += added
    
    return total, results


print('='*60)
print('STAGE 1 CONSTRAINT COUNT COMPARISON - 2026')
print('='*60)
print('(Using model.Proto().constraints length - accurate method)')

data = load_data(2026)

# Standard constraints
print('\n=== ORIGINAL CONSTRAINTS ===')
stage1 = STAGES['stage1_required']
print(f"Stage: {stage1['name']}")
print(f"Constraints ({len(stage1['constraints'])} classes):")

model = cp_model.CpModel()
X, Y, conflicts, unavailable = generate_X(model, data)
data['unavailable_games'] = unavailable
data['team_conflicts'] = conflicts
data['penalties'] = {}

decision_vars = len([k for k in X.keys()])
print(f"\nDecision variables (X): {decision_vars:,}")
print(f"Model variables before constraints: {len(model.Proto().variables):,}")
print(f"Model constraints before: {len(model.Proto().constraints):,}")

total_original, results_original = count_constraints_applied(
    model, stage1['constraints'], X, data
)

print(f"\nConstraints applied:")
for name, count in results_original:
    print(f"  {name}: {count:,}")

print(f"\nTOTAL ORIGINAL: {total_original:,} constraints")
print(f"Model variables after: {len(model.Proto().variables):,}")
print(f"Model constraints after: {len(model.Proto().constraints):,}")

# AI constraints  
print('\n\n=== AI-ENHANCED CONSTRAINTS ===')
stage1_ai = STAGES_AI['stage1_required']
print(f"Stage: {stage1_ai['name']}")
print(f"Constraints ({len(stage1_ai['constraints'])} classes):")

model_ai = cp_model.CpModel()
X_ai, Y_ai, conflicts_ai, unavailable_ai = generate_X(model_ai, data)
data['unavailable_games'] = unavailable_ai
data['team_conflicts'] = conflicts_ai
data['penalties'] = {}

print(f"\nDecision variables (X): {len(X_ai):,}")
print(f"Model variables before constraints: {len(model_ai.Proto().variables):,}")
print(f"Model constraints before: {len(model_ai.Proto().constraints):,}")

total_ai, results_ai = count_constraints_applied(
    model_ai, stage1_ai['constraints'], X_ai, data
)

print(f"\nConstraints applied:")
for name, count in results_ai:
    print(f"  {name}: {count:,}")

print(f"\nTOTAL AI: {total_ai:,} constraints")
print(f"Model variables after: {len(model_ai.Proto().variables):,}")
print(f"Model constraints after: {len(model_ai.Proto().constraints):,}")

print('\n' + '='*60)
print('COMPARISON')
print('='*60)
diff = total_original - total_ai
pct = (diff / total_original * 100) if total_original > 0 else 0
print(f"Original: {total_original:,} constraints")
print(f"AI:       {total_ai:,} constraints")
print(f"Difference: {diff:,} ({pct:.1f}% {'fewer' if diff > 0 else 'more'} with AI)")

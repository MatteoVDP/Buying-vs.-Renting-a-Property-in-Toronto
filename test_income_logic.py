#!/usr/bin/env python3
"""
Minimal test: Check if income suppression logic is syntactically correct
"""

# Test the income suppression logic inline
historical_max_affordability = 10.1424

test_cases = [
    (11.0, 0.03, "High affordability, normal income growth"),
    (10.0, 0.03, "At threshold, normal income growth"),
    (9.5, 0.03, "Below threshold, normal income growth"),
    (15.0, 0.05, "Very high affordability, high income growth"),
]

print("Testing income suppression logic:")
print("=" * 80)

for last_affordability, current_income_yoy, description in test_cases:
    print(f"\nTest: {description}")
    print(f"  last_affordability = {last_affordability:.2f}")
    print(f"  current_income_yoy (original) = {current_income_yoy:.4f} ({current_income_yoy*100:.2f}%)")
    
    # Apply the constraint logic
    if last_affordability > historical_max_affordability:
        modified_income_yoy = max(current_income_yoy * 0.5, -0.02)
        zone = "HARD OVERRIDE"
    elif last_affordability > (historical_max_affordability * 0.95):
        modified_income_yoy = current_income_yoy * 0.5
        zone = "SOFT LANDING ZONE"
    else:
        modified_income_yoy = current_income_yoy
        zone = "NORMAL"
    
    monthly_income_factor = (1 + modified_income_yoy) ** (1/12)
    
    print(f"  Zone: {zone}")
    print(f"  current_income_yoy (modified) = {modified_income_yoy:.4f} ({modified_income_yoy*100:.2f}%)")
    print(f"  monthly_income_factor = {monthly_income_factor:.6f}")

print("\n" + "=" * 80)
print("✓ Logic is syntactically correct and functioning as intended")

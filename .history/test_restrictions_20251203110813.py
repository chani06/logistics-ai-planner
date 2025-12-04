# -*- coding: utf-8 -*-
"""Test branch vehicle restrictions learning"""
import pandas as pd
import sys

# Import function
sys.path.insert(0, '.')
from app import PUNTHAI_PATTERNS, get_max_vehicle_for_branch, check_branch_vehicle_compatibility

print("="*60)
print("Branch Vehicle Restrictions Test")
print("="*60)

restrictions = PUNTHAI_PATTERNS.get('branch_restrictions', {})
print(f"\nBranch restrictions loaded: {len(restrictions)} branches")

if restrictions:
    # Count by restriction type
    strict = [b for b, r in restrictions.items() if len(r['allowed']) == 1]
    flexible = [b for b, r in restrictions.items() if len(r['allowed']) > 1]
    
    print(f"- Strict (1 vehicle only): {len(strict)}")
    print(f"- Flexible (multiple): {len(flexible)}")
    
    # Count by vehicle type
    vehicle_counts = {}
    for r in restrictions.values():
        max_v = r['max_vehicle']
        vehicle_counts[max_v] = vehicle_counts.get(max_v, 0) + 1
    
    print("\nMax vehicle distribution:")
    for v, count in sorted(vehicle_counts.items()):
        print(f"  {v}: {count} branches")
    
    # Show sample strict restrictions
    print("\nSample strict restrictions:")
    for branch_code in list(strict)[:10]:
        info = restrictions[branch_code]
        print(f"  {branch_code}: {info['allowed']} (max: {info['max_vehicle']})")
    
    # Test compatibility functions
    print("\n" + "="*60)
    print("Testing compatibility functions:")
    print("="*60)
    
    # Test a 4W-only branch
    for branch_code in list(strict)[:3]:
        info = restrictions[branch_code]
        if info['max_vehicle'] == '4W':
            print(f"\nBranch {branch_code} (4W-only):")
            print(f"  Can use 4W? {check_branch_vehicle_compatibility(branch_code, '4W')}")
            print(f"  Can use JB? {check_branch_vehicle_compatibility(branch_code, 'JB')}")
            print(f"  Can use 6W? {check_branch_vehicle_compatibility(branch_code, '6W')}")
            print(f"  Max vehicle: {get_max_vehicle_for_branch(branch_code)}")
            break
    
    # Test a 6W-only branch
    for branch_code in list(strict)[:10]:
        info = restrictions[branch_code]
        if info['max_vehicle'] == '6W':
            print(f"\nBranch {branch_code} (6W-only):")
            print(f"  Can use 4W? {check_branch_vehicle_compatibility(branch_code, '4W')}")
            print(f"  Can use JB? {check_branch_vehicle_compatibility(branch_code, 'JB')}")
            print(f"  Can use 6W? {check_branch_vehicle_compatibility(branch_code, '6W')}")
            print(f"  Max vehicle: {get_max_vehicle_for_branch(branch_code)}")
            break
    
    # Test a flexible branch
    for branch_code in list(flexible)[:3]:
        info = restrictions[branch_code]
        print(f"\nBranch {branch_code} (flexible - {info['allowed']}):")
        print(f"  Can use 4W? {check_branch_vehicle_compatibility(branch_code, '4W')}")
        print(f"  Can use JB? {check_branch_vehicle_compatibility(branch_code, 'JB')}")
        print(f"  Can use 6W? {check_branch_vehicle_compatibility(branch_code, '6W')}")
        print(f"  Max vehicle: {get_max_vehicle_for_branch(branch_code)}")
        break

else:
    print("ERROR: No branch restrictions loaded!")

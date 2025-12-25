#!/usr/bin/env python
"""
Test script to verify separate P_clean and P_dirty generation.
"""
import sys
sys.path.insert(0, '/mnt/data/welkinni/table_det')

from agentic_error_detector.legislator import DualLegislator

print("="*80)
print("TESTING SEPARATE P_CLEAN AND P_DIRTY GENERATION")
print("="*80)
print()

# Create DualLegislator instance
legislator = DualLegislator()

# Create test metadata
test_metadata = {
    'type': 'categorical',
    'sample_values': ['diabetes', 'hypertension', 'asthma', 'cancer', 'flu'],
    'top_values': {'diabetes': 100, 'hypertension': 80, 'asthma': 60},
    'unique_count': 10,
    'null_count': 5
}

print("Test metadata:")
print(f"  Type: {test_metadata['type']}")
print(f"  Sample values: {test_metadata['sample_values']}")
print()

# Test P_clean generation
print("="*80)
print("TEST 1: Generate P_clean rule separately")
print("="*80)
try:
    p_clean = legislator.generate_p_clean_rule('test_column', test_metadata)
    print(f"✓ P_clean generated successfully:")
    print(f"  {p_clean[:100]}...")
except Exception as e:
    print(f"✗ Error generating P_clean: {e}")
    import traceback
    traceback.print_exc()

print()

# Test P_dirty generation
print("="*80)
print("TEST 2: Generate P_dirty rule separately")
print("="*80)
try:
    p_dirty = legislator.generate_p_dirty_rule('test_column', test_metadata)
    print(f"✓ P_dirty generated successfully:")
    print(f"  {p_dirty[:100]}...")
except Exception as e:
    print(f"✗ Error generating P_dirty: {e}")
    import traceback
    traceback.print_exc()

print()

# Test dual rules generation
print("="*80)
print("TEST 3: Generate dual rules (should call both separately)")
print("="*80)
try:
    dual_rules = legislator.generate_dual_rules('test_column', test_metadata)
    if dual_rules:
        agent_name, clean_rule, dirty_rule = dual_rules[0]
        print(f"✓ Dual rules generated successfully:")
        print(f"  Agent: {agent_name}")
        print(f"  P_clean: {clean_rule[:100]}...")
        print(f"  P_dirty: {dirty_rule[:100]}...")
    else:
        print("✗ No dual rules generated")
except Exception as e:
    print(f"✗ Error generating dual rules: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*80)
print("TEST COMPLETE")
print("="*80)

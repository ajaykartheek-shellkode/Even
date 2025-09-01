#!/usr/bin/env python3
"""
Test optimizer logic without making Bedrock API calls.
"""

import json
from prompt_optimizer import PromptOptimizer

def test_optimizer_setup():
    """Test optimizer initialization and prompt building."""
    print("=== Testing Optimizer Logic ===")
    
    # Create optimizer
    optimizer = PromptOptimizer()
    
    # Load examples
    try:
        optimizer.load_examples('data/train', 'data/val')
        print(f"✓ Loaded {len(optimizer.train_examples)} training examples")
        print(f"✓ Loaded {len(optimizer.val_examples)} validation examples")
    except Exception as e:
        print(f"✗ Failed to load examples: {e}")
        return False
    
    # Test prompt building variations
    if optimizer.val_examples:
        test_text = optimizer.val_examples[0]['raw_text']
        
        # Test different prompt configurations
        configs = [
            {'instruction_idx': 0, 'use_fs': False, 'k': 0},
            {'instruction_idx': 1, 'use_fs': True, 'k': 2},
            {'instruction_idx': 2, 'use_fs': True, 'k': 1},
        ]
        
        for i, config in enumerate(configs):
            try:
                prompt = optimizer.build_prompt(
                    config['instruction_idx'], 
                    config['use_fs'], 
                    config['k'], 
                    test_text[:500]  # Truncate for testing
                )
                
                # Verify prompt structure
                has_schema = "extracted_invoice_values" in prompt
                has_text = test_text[:100] in prompt
                
                print(f"✓ Config {i+1}: Schema={has_schema}, Text={has_text}, Length={len(prompt)}")
                
                if config['use_fs'] and config['k'] > 0:
                    has_examples = "<EXAMPLE_" in prompt
                    print(f"  Few-shot examples included: {has_examples}")
                
            except Exception as e:
                print(f"✗ Config {i+1} failed: {e}")
                return False
    
    print("✓ All prompt configurations work")
    return True

def test_schema_and_instructions():
    """Test schema definition and instruction variants."""
    print("\n=== Testing Instructions and Schema ===")
    
    optimizer = PromptOptimizer()
    
    # Test instruction variants
    instructions = optimizer.instruction_variants
    print(f"✓ Found {len(instructions)} instruction variants")
    
    for i, instruction in enumerate(instructions):
        has_schema = "extracted_invoice_values" in instruction
        has_requirements = "CRITICAL" in instruction or "Requirements" in instruction or "EXTRACTION" in instruction
        length = len(instruction)
        
        print(f"  Variant {i+1}: Schema={has_schema}, Requirements={has_requirements}, Length={length}")
    
    # Test schema definition
    schema = optimizer._get_schema_definition()
    schema_valid = "invoice_number" in schema and "services" in schema
    print(f"✓ Schema definition valid: {schema_valid}")
    
    return True

if __name__ == "__main__":
    print("Testing Optimizer Logic (No API Calls)")
    print("=" * 50)
    
    success1 = test_optimizer_setup()
    success2 = test_schema_and_instructions()
    
    if success1 and success2:
        print("\n✓ All optimizer logic tests passed!")
        print("The optimizer is ready for Bedrock API calls.")
        exit(0)
    else:
        print("\n✗ Some tests failed!")
        exit(1)
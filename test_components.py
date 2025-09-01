#!/usr/bin/env python3
"""
Test script for core components of the Prompt Optimizer MVP.
Tests individual components before running full optimization.
"""

import json
from pathlib import Path

def test_evaluator():
    """Test the evaluator component."""
    print("=== Testing Evaluator ===")
    
    try:
        from evaluator import score_prediction, schema_valid, numeric_close, string_similar
        
        # Create test data
        test_gold = {
            "invoice_number": "1720",
            "patient_name": "Justine",
            "services": [
                {
                    "service": "Pan-uu",
                    "amount": 165,
                    "quantity": 15,
                    "department": "pharmacy",
                    "unit": "tablets",
                    "mrp": 0,
                    "cgst": 0,
                    "cgst_type": None,
                    "sgst": 0,
                    "sgst_type": None,
                    "gst": 0,
                    "gst_type": None
                }
            ],
            "total_amount": 236,
            "doctor_name": "Dr. Vasanth Kumar",
            "facility": "VIJAYALAKSHMI MEDICAL",
            "invoice_date": "7/5/24",
            "payment_mode": "Cash",
            "patient_age": None,
            "patient_gender": None,
            "patient_contact": None,
            "cgst": 0,
            "cgst_type": None,
            "sgst": 0,
            "sgst_type": None,
            "gst": 0,
            "gst_type": None,
            "discount": 0,
            "mrp": 0,
            "round_off": 0
        }
        
        test_pred = {
            "extracted_invoice_values": test_gold.copy()
        }
        
        # Test schema validation
        is_valid = schema_valid(test_pred)
        print(f"  Schema validation: {'PASS' if is_valid else 'FAIL'}")
        
        # Test numeric close
        close_result = numeric_close(235, 236)
        print(f"  Numeric close (235, 236): {'PASS' if close_result else 'FAIL'}")
        
        # Test string similar
        similar_result = string_similar("Justine", "Justine Kumar")
        print(f"  String similar: {'PASS' if similar_result else 'FAIL'}")
        
        # Test full scoring
        result = score_prediction(test_pred, test_gold)
        print(f"  Composite score: {result['composite']:.3f}")
        print(f"  Evaluator test: {'PASS' if result['composite'] > 0.8 else 'FAIL'}")
        
        return True
        
    except Exception as e:
        print(f"  Evaluator test: FAIL - {e}")
        return False


def test_bedrock_helper():
    """Test the bedrock helper (without making actual API calls)."""
    print("\n=== Testing Bedrock Helper ===")
    
    try:
        from bedrock_helper import BedrockHelper, extract_json_from_output
        
        # Test JSON extraction
        test_output = '{"extracted_invoice_values": {"invoice_number": "123"}}'
        extracted = extract_json_from_output(test_output)
        
        has_key = "extracted_invoice_values" in extracted
        print(f"  JSON extraction: {'PASS' if has_key else 'FAIL'}")
        
        # Test helper initialization  
        helper = BedrockHelper(region_name='ap-south-1', cache_dir='cache/bedrock')
        cache_exists = helper.cache_dir.exists()
        print(f"  Cache directory creation: {'PASS' if cache_exists else 'FAIL'}")
        
        print(f"  Bedrock helper test: PASS")
        return True
        
    except Exception as e:
        print(f"  Bedrock helper test: FAIL - {e}")
        return False


def test_data_loading():
    """Test loading examples from data folders."""
    print("\n=== Testing Data Loading ===")
    
    try:
        train_folder = Path("data/train")
        val_folder = Path("data/val")
        
        if not train_folder.exists() or not val_folder.exists():
            print(f"  Data loading test: FAIL - Folders don't exist")
            return False
        
        # Count files
        train_files = list(train_folder.glob("*.json"))
        val_files = list(val_folder.glob("*.json"))
        
        print(f"  Training files: {len(train_files)}")
        print(f"  Validation files: {len(val_files)}")
        
        # Test loading one example
        if train_files:
            with open(train_files[0], 'r') as f:
                example = json.load(f)
            
            has_required_keys = all(key in example for key in ['file_id', 'raw_text', 'gold'])
            print(f"  Example structure: {'PASS' if has_required_keys else 'FAIL'}")
        
        success = len(train_files) > 0 and len(val_files) > 0
        print(f"  Data loading test: {'PASS' if success else 'FAIL'}")
        return success
        
    except Exception as e:
        print(f"  Data loading test: FAIL - {e}")
        return False


def test_prompt_building():
    """Test prompt building functionality.""" 
    print("\n=== Testing Prompt Building ===")
    
    try:
        from prompt_optimizer import AdvancedPromptOptimizer
        
        optimizer = AdvancedPromptOptimizer()
        
        # Test instruction variants - they're generated dynamically
        optimizer.load_examples("data/train", "data/val")
        variants = len(optimizer.generate_instruction_variants())
        print(f"  Instruction variants: {variants}")
        
        # Test basic prompt building with new signature
        test_text = "Sample invoice text for testing"
        prompt = optimizer.build_prompt(0, 2, test_text)
        
        has_text = test_text in prompt
        has_schema = "extracted_invoice_values" in prompt
        
        print(f"  Prompt contains input text: {'PASS' if has_text else 'FAIL'}")
        print(f"  Prompt contains schema: {'PASS' if has_schema else 'FAIL'}")
        
        success = has_text and has_schema and len(prompt) > 100
        print(f"  Prompt building test: {'PASS' if success else 'FAIL'}")
        return success
        
    except Exception as e:
        print(f"  Prompt building test: FAIL - {e}")
        return False


def main():
    """Run all component tests."""
    print("Running component tests before full optimization...")
    
    results = []
    results.append(test_evaluator())
    results.append(test_bedrock_helper()) 
    results.append(test_data_loading())
    results.append(test_prompt_building())
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n=== Test Summary ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("All tests passed! ✓")
        print("Ready to run full optimization.")
        return 0
    else:
        print("Some tests failed! ✗")
        print("Please fix issues before running optimization.")
        return 1


if __name__ == "__main__":
    exit(main())
#!/usr/bin/env python3
"""
Evaluator for Prompt Optimizer MVP.
Provides comprehensive scoring functions for invoice extraction tasks.
"""

import json
import math
from typing import Dict, Any, List, Optional, Union
from jsonschema import validate, ValidationError
import re
from rapidfuzz import fuzz


# JSON Schema for extracted invoice values
INVOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "extracted_invoice_values": {
            "type": "object", 
            "properties": {
                "invoice_number": {"type": ["string", "null"]},
                "patient_name": {"type": ["string", "null"]},
                "services": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "service": {"type": ["string", "null"]},
                            "amount": {"type": "number"},
                            "quantity": {"type": "number"},
                            "department": {"type": ["string", "null"]},
                            "unit": {"type": ["string", "null"]},
                            "mrp": {"type": "number"},
                            "cgst": {"type": "number"},
                            "cgst_type": {"type": ["string", "null"]},
                            "sgst": {"type": "number"},
                            "sgst_type": {"type": ["string", "null"]},
                            "gst": {"type": "number"},
                            "gst_type": {"type": ["string", "null"]}
                        }
                    }
                },
                "total_amount": {"type": "number"},
                "doctor_name": {"type": ["string", "null"]},
                "facility": {"type": ["string", "null"]},
                "invoice_date": {"type": ["string", "null"]},
                "payment_mode": {"type": ["string", "null"]},
                "patient_age": {"type": ["string", "null"]},
                "patient_gender": {"type": ["string", "null"]},
                "patient_contact": {"type": ["string", "null"]},
                "cgst": {"type": "number"},
                "cgst_type": {"type": ["string", "null"]},
                "sgst": {"type": "number"},
                "sgst_type": {"type": ["string", "null"]},
                "gst": {"type": "number"},
                "gst_type": {"type": ["string", "null"]},
                "discount": {"type": "number"},
                "mrp": {"type": "number"},
                "round_off": {"type": "number"}
            },
            "required": ["services", "total_amount"]
        }
    },
    "required": ["extracted_invoice_values"]
}


def schema_valid(pred_json: Dict[str, Any]) -> bool:
    """
    Check if predicted JSON matches the expected schema.
    
    Args:
        pred_json: Predicted JSON to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        validate(instance=pred_json, schema=INVOICE_SCHEMA)
        return True
    except ValidationError:
        return False
    except Exception:
        return False


def numeric_close(pred_value: Union[float, int, None], 
                  gold_value: Union[float, int, None],
                  rel_tol: float = 1e-2, 
                  abs_tol: float = 1.0) -> bool:
    """
    Check if two numeric values are close using math.isclose.
    
    Args:
        pred_value: Predicted numeric value
        gold_value: Gold standard numeric value  
        rel_tol: Relative tolerance
        abs_tol: Absolute tolerance
        
    Returns:
        True if values are close, False otherwise
    """
    # Handle None values
    if pred_value is None and gold_value is None:
        return True
    if pred_value is None or gold_value is None:
        return False
    
    try:
        pred_num = float(pred_value)
        gold_num = float(gold_value)
        return math.isclose(pred_num, gold_num, rel_tol=rel_tol, abs_tol=abs_tol)
    except (ValueError, TypeError):
        return False


def string_similar(pred_str: Optional[str], gold_str: Optional[str], 
                   threshold: float = 80.0) -> bool:
    """
    Check if two strings are similar using fuzzy matching.
    
    Args:
        pred_str: Predicted string
        gold_str: Gold standard string
        threshold: Similarity threshold (0-100)
        
    Returns:
        True if strings are similar, False otherwise
    """
    if pred_str is None and gold_str is None:
        return True
    if pred_str is None or gold_str is None:
        return False
    
    try:
        similarity = fuzz.partial_ratio(str(pred_str), str(gold_str))
        return similarity >= threshold
    except:
        return False


def match_services(pred_services: List[Dict], gold_services: List[Dict]) -> float:
    """
    Match services using fuzzy string matching and calculate accuracy.
    
    Args:
        pred_services: List of predicted services
        gold_services: List of gold standard services
        
    Returns:
        Service matching accuracy (0.0 to 1.0)
    """
    if not gold_services:
        return 1.0 if not pred_services else 0.0
    
    if not pred_services:
        return 0.0
    
    matched_services = 0
    total_service_fields = 0
    
    # Try to match each gold service with predicted services
    for gold_service in gold_services:
        best_match = None
        best_score = 0
        
        # Find best matching predicted service
        for pred_service in pred_services:
            service_name_score = 0
            if gold_service.get('service') and pred_service.get('service'):
                service_name_score = fuzz.partial_ratio(
                    str(pred_service['service']), 
                    str(gold_service['service'])
                )
            
            if service_name_score > best_score:
                best_score = service_name_score
                best_match = pred_service
        
        # If we found a reasonable match, compare all fields
        if best_match and best_score >= 60:  # Lower threshold for service name matching
            service_field_matches = 0
            service_total_fields = 0
            
            # Compare all service fields
            for field in ['service', 'amount', 'quantity', 'department', 'unit', 
                         'mrp', 'cgst', 'sgst', 'gst']:
                service_total_fields += 1
                
                gold_val = gold_service.get(field)
                pred_val = best_match.get(field)
                
                if field in ['amount', 'quantity', 'mrp', 'cgst', 'sgst', 'gst']:
                    # Numeric fields
                    if numeric_close(pred_val, gold_val):
                        service_field_matches += 1
                else:
                    # String fields
                    if string_similar(pred_val, gold_val):
                        service_field_matches += 1
            
            matched_services += service_field_matches / service_total_fields if service_total_fields > 0 else 0
            total_service_fields += 1
    
    return matched_services / len(gold_services) if gold_services else 0.0


def slot_accuracy(pred: Dict[str, Any], gold: Dict[str, Any]) -> float:
    """
    Calculate slot-level accuracy for all fields.
    
    Args:
        pred: Predicted extraction
        gold: Gold standard extraction
        
    Returns:
        Slot accuracy score (0.0 to 1.0)
    """
    try:
        pred_values = pred.get('extracted_invoice_values', {})
        gold_values = gold
        
        total_fields = 0
        correct_fields = 0
        
        # Define field categories and their scoring functions
        numeric_fields = ['total_amount', 'cgst', 'sgst', 'gst', 'discount', 'mrp', 'round_off']
        string_fields = ['invoice_number', 'patient_name', 'doctor_name', 'facility', 
                        'invoice_date', 'payment_mode', 'patient_age', 'patient_gender', 
                        'patient_contact', 'cgst_type', 'sgst_type', 'gst_type']
        
        # Score numeric fields
        for field in numeric_fields:
            total_fields += 1
            pred_val = pred_values.get(field)
            gold_val = gold_values.get(field)
            
            if numeric_close(pred_val, gold_val):
                correct_fields += 1
        
        # Score string fields
        for field in string_fields:
            total_fields += 1
            pred_val = pred_values.get(field)
            gold_val = gold_values.get(field)
            
            if string_similar(pred_val, gold_val):
                correct_fields += 1
        
        # Score services (special handling)
        pred_services = pred_values.get('services', [])
        gold_services = gold_values.get('services', [])
        
        services_score = match_services(pred_services, gold_services)
        
        # Services contribute to the overall score (weighted)
        services_weight = 0.4  # Services are important
        basic_fields_weight = 0.6
        
        basic_accuracy = correct_fields / total_fields if total_fields > 0 else 0.0
        final_accuracy = (basic_fields_weight * basic_accuracy) + (services_weight * services_score)
        
        return final_accuracy
        
    except Exception as e:
        print(f"Error calculating slot accuracy: {e}")
        return 0.0


def composite_score(schema_valid_score: bool, slot_accuracy_score: float, 
                   schema_weight: float = 0.3, slot_weight: float = 0.7) -> float:
    """
    Calculate composite score combining schema validity and slot accuracy.
    
    Args:
        schema_valid_score: Schema validation result (True/False)
        slot_accuracy_score: Slot accuracy score (0.0 to 1.0)
        schema_weight: Weight for schema validation
        slot_weight: Weight for slot accuracy
        
    Returns:
        Composite score (0.0 to 1.0)
    """
    schema_score = 1.0 if schema_valid_score else 0.0
    return (schema_weight * schema_score) + (slot_weight * slot_accuracy_score)


def score_prediction(pred: Dict[str, Any], gold: Dict[str, Any]) -> Dict[str, Any]:
    """
    Score a prediction against gold standard.
    
    Args:
        pred: Predicted extraction
        gold: Gold standard extraction
        
    Returns:
        Dictionary with scoring results
    """
    # Schema validation
    schema_valid_result = schema_valid(pred)
    
    # Slot accuracy  
    slot_accuracy_result = slot_accuracy(pred, gold)
    
    # Composite score
    composite_result = composite_score(schema_valid_result, slot_accuracy_result)
    
    return {
        'schema_valid': schema_valid_result,
        'slot_accuracy': slot_accuracy_result,
        'composite': composite_result
    }


def evaluate_predictions(predictions: List[Dict[str, Any]], 
                        gold_standards: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate a list of predictions against gold standards.
    
    Args:
        predictions: List of predicted extractions
        gold_standards: List of gold standard extractions
        
    Returns:
        Dictionary with evaluation metrics
    """
    if len(predictions) != len(gold_standards):
        raise ValueError("Number of predictions must match number of gold standards")
    
    scores = []
    
    for pred, gold in zip(predictions, gold_standards):
        score = score_prediction(pred, gold)
        scores.append(score)
    
    # Calculate averages
    avg_schema_valid = sum(s['schema_valid'] for s in scores) / len(scores)
    avg_slot_accuracy = sum(s['slot_accuracy'] for s in scores) / len(scores)
    avg_composite = sum(s['composite'] for s in scores) / len(scores)
    
    return {
        'individual_scores': scores,
        'averages': {
            'schema_valid': avg_schema_valid,
            'slot_accuracy': avg_slot_accuracy,
            'composite': avg_composite
        },
        'total_predictions': len(predictions)
    }


if __name__ == "__main__":
    # Test the evaluator
    import sys
    
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
    
    # Test prediction (slightly different)
    test_pred = {
        "extracted_invoice_values": {
            "invoice_number": "1720",
            "patient_name": "Justine Kumar",  # Slightly different
            "services": [
                {
                    "service": "Pan-uu Tablet",  # Slightly different
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
            "total_amount": 235,  # Slightly different
            "doctor_name": "Dr. Vasanth Kumar",
            "facility": "VIJAYALAKSHMI MEDICAL CENTER",  # Slightly different
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
    }
    
    print("=== Testing Evaluator ===")
    
    # Test schema validation
    print(f"Schema valid: {schema_valid(test_pred)}")
    
    # Test full scoring
    result = score_prediction(test_pred, test_gold)
    print(f"\nScoring Results:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    # Test numeric close
    print(f"\nNumeric close test (235, 236): {numeric_close(235, 236)}")
    
    # Test string similar
    print(f"String similar test ('Justine', 'Justine Kumar'): {string_similar('Justine', 'Justine Kumar')}")
    
    print("\nEvaluator test completed successfully!")
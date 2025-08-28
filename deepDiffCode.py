import json
import logging
from deepdiff import DeepDiff
from typing import Dict, Any, Union, Tuple

def validate_json(expected_json_str: Union[str, dict], predicted_json_str: Union[str, dict]) -> Dict[str, Any]:
    """
    Comprehensive JSON validation function using DeepDiff.
    
    Args:
        expected_json_str: Expected JSON (string or dict)
        predicted_json_str: Predicted JSON (string or dict) 
        
    Returns:
        Dictionary with validation results including differences and feedback
    """
    try:
        if isinstance(expected_json_str, str):
            expected = json.loads(expected_json_str)
        else:
            expected = expected_json_str
            
        if isinstance(predicted_json_str, str):
            predicted = json.loads(predicted_json_str)
        else:
            predicted = predicted_json_str
            
    except json.JSONDecodeError as e:
        return {
            "valid": False,
            "error": f"Invalid JSON format: {e}",
            "differences": {},
            "differences_pretty": f"JSON parsing error: {e}",
            "score": 0.0
        }
    except Exception as e:
        return {
            "valid": False,
            "error": f"Unexpected error during JSON validation: {e}",
            "differences": {},
            "differences_pretty": f"Validation error: {e}",
            "score": 0.0
        }

    try:
        diff = DeepDiff(expected, predicted, ignore_order=True)
        
        is_valid = len(diff) == 0
        score = 1.0 if is_valid else calculate_similarity_score(diff, expected, predicted)
        
        differences_pretty = "JSONs match perfectly." if is_valid else generate_human_readable_diff(diff)
        
        return {
            "valid": is_valid,
            "differences": diff.to_dict() if diff else {},
            "differences_pretty": differences_pretty,
            "score": score
        }
        
    except Exception as e:
        return {
            "valid": False,
            "error": f"Error during DeepDiff comparison: {e}",
            "differences": {},
            "differences_pretty": f"Comparison error: {e}",
            "score": 0.0
        }

def calculate_similarity_score(diff: DeepDiff, expected: dict, predicted: dict) -> float:
    """
    Calculate a similarity score based on the differences.
    Returns a score between 0 and 1, where 1 is perfect match.
    """
    if not diff:
        return 1.0
        
    total_fields = count_total_fields(expected)
    if total_fields == 0:
        return 1.0
    
    error_weight = 0
    
    if 'dictionary_item_added' in diff:
        error_weight += len(diff['dictionary_item_added']) * 0.5
    if 'dictionary_item_removed' in diff:
        error_weight += len(diff['dictionary_item_removed']) * 1.0
    if 'values_changed' in diff:
        error_weight += len(diff['values_changed']) * 0.8
    if 'type_changes' in diff:
        error_weight += len(diff['type_changes']) * 1.0
    if 'iterable_item_added' in diff:
        error_weight += len(diff['iterable_item_added']) * 0.3
    if 'iterable_item_removed' in diff:
        error_weight += len(diff['iterable_item_removed']) * 0.7
        
    similarity = max(0.0, 1.0 - (error_weight / total_fields))
    return similarity

def count_total_fields(obj: Any, depth: int = 0, max_depth: int = 10) -> int:
    """Recursively count total fields in a nested structure."""
    if depth > max_depth:
        return 1
        
    if isinstance(obj, dict):
        return sum(count_total_fields(v, depth + 1, max_depth) for v in obj.values()) + len(obj)
    elif isinstance(obj, (list, tuple)):
        return sum(count_total_fields(item, depth + 1, max_depth) for item in obj) + len(obj)
    else:
        return 1

def generate_human_readable_diff(diff: DeepDiff) -> str:
    """Generate human-readable explanation of differences."""
    explanations = []
    
    if 'dictionary_item_added' in diff:
        added = diff['dictionary_item_added']
        for path in added:
            explanations.append(f"Added field: {path}")
    
    if 'dictionary_item_removed' in diff:
        removed = diff['dictionary_item_removed']
        for path in removed:
            explanations.append(f"Missing field: {path}")
    
    if 'values_changed' in diff:
        changed = diff['values_changed']
        for path, change in changed.items():
            old_val = change.get('old_value', 'N/A')
            new_val = change.get('new_value', 'N/A')
            explanations.append(f"Changed {path}: '{old_val}' -> '{new_val}'")
    
    if 'type_changes' in diff:
        type_changed = diff['type_changes']
        for path, change in type_changed.items():
            old_type = change.get('old_type', 'N/A')
            new_type = change.get('new_type', 'N/A')
            explanations.append(f"Type changed {path}: {old_type} -> {new_type}")
    
    if 'iterable_item_added' in diff:
        added = diff['iterable_item_added']
        for path in added:
            explanations.append(f"Added array item: {path}")
    
    if 'iterable_item_removed' in diff:
        removed = diff['iterable_item_removed']
        for path in removed:
            explanations.append(f"Removed array item: {path}")
    
    return "\n".join(explanations) if explanations else "Unknown differences detected"

def compute_diff(expected: dict, predicted: dict) -> DeepDiff:
    """Return DeepDiff differences between two JSON dicts."""
    return DeepDiff(expected, predicted, ignore_order=True)

def diff_score(expected: dict, predicted: dict) -> float:
    """Return similarity score (1 = perfect match, 0 = completely different)."""
    validation_result = validate_json(expected, predicted)
    return validation_result["score"]

def make_feedback(expected: dict, predicted: dict) -> str:
    """Return human-readable explanation of differences."""
    validation_result = validate_json(expected, predicted)
    return validation_result["differences_pretty"]
import json
import glob
from pathlib import Path
from deepdiff import DeepDiff

def validate_json(expected_json_str, predicted_json_str):
    try:
        expected = json.loads(expected_json_str)
        predicted = json.loads(predicted_json_str)
    except Exception as e:
        return {"valid": False, "error": f"Invalid JSON: {e}"}

    diff = DeepDiff(expected, predicted, ignore_order=True)

    feedback = []
    for category, changes in diff.items():
        for path, change in (changes.items() if isinstance(changes, dict) else enumerate(changes)):
            feedback.append(f"{category}: {path} → {change}")

    result = {
        "valid": len(diff) == 0,
        "differences": diff.to_dict(),
    }

    if result["valid"]:
        result["feedback"] = "Both the JSONs match exactly"
    else:
        result["feedback"] = "The output JSON does not match the expected JSON.\n- " + "\n- ".join(feedback)

    return result

def get_json_files_from_folder(folder_path):
    """Get all JSON files from a folder"""
    folder_path = Path(folder_path)
    if not folder_path.exists():
        print(f"Warning: Folder does not exist: {folder_path}")
        return []
    
    json_files = glob.glob(str(folder_path / "*.json"))
    return sorted(json_files)

def load_json_file(file_path):
    """Load JSON content from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def find_matching_files(expected_folder, extracted_folder):
    """Find matching JSON files between expected and extracted folders"""
    expected_files = get_json_files_from_folder(expected_folder)
    extracted_files = get_json_files_from_folder(extracted_folder)
    
    expected_basenames = {Path(f).name: f for f in expected_files}
    extracted_basenames = {Path(f).name: f for f in extracted_files}
    
    matching_pairs = []
    for filename in expected_basenames:
        if filename in extracted_basenames:
            matching_pairs.append({
                'filename': filename,
                'expected_path': expected_basenames[filename],
                'extracted_path': extracted_basenames[filename]
            })
        else:
            print(f"Warning: No matching extracted file for {filename}")
    
    for filename in extracted_basenames:
        if filename not in expected_basenames:
            print(f"Warning: No matching expected file for {filename}")
    
    return matching_pairs

def compare_all_files(expected_folder, extracted_folder):
    """Compare all matching JSON files between two folders"""
    print(f"Comparing files between:")
    print(f"Expected folder: {expected_folder}")
    print(f"Extracted folder: {extracted_folder}")
    print("="*80)
    
    matching_pairs = find_matching_files(expected_folder, extracted_folder)
    
    if not matching_pairs:
        print("No matching files found!")
        return []
    
    results = []
    total_matches = 0
    total_files = len(matching_pairs)
    
    for pair in matching_pairs:
        filename = pair['filename']
        expected_path = pair['expected_path']
        extracted_path = pair['extracted_path']
        
        print(f"\nComparing: {filename}")
        print("-" * 50)
        
        # Load JSON content
        expected_json_str = load_json_file(expected_path)
        extracted_json_str = load_json_file(extracted_path)
        
        if expected_json_str is None or extracted_json_str is None:
            result = {
                "filename": filename,
                "valid": False,
                "error": "Failed to load one or both files",
                "expected_path": expected_path,
                "extracted_path": extracted_path
            }
        else:
            # Compare using existing validation function
            result = validate_json(expected_json_str, extracted_json_str)
            result["filename"] = filename
            result["expected_path"] = expected_path
            result["extracted_path"] = extracted_path
        
        results.append(result)
        
        # Print result for this file
        if result.get("valid", False):
            print(f"MATCH: {filename}")
            total_matches += 1
        else:
            print(f"MISMATCH: {filename}")
            if "error" in result:
                print(f"   Error: {result['error']}")
            else:
                print(f"   Feedback: {result['feedback']}")
    
    # Print summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"Total files compared: {total_files}")
    print(f"Exact matches: {total_matches}")
    print(f"Mismatches: {total_files - total_matches}")
    print(f"Match rate: {(total_matches/total_files)*100:.1f}%" if total_files > 0 else "0.0%")
    
    return results
def main():
    # Configure folder paths
    expected_folder = "expected-output"
    extracted_folder = "extracted-output"
    
    print("🔍 Starting dynamic JSON comparison...")
    print("=" * 80)
    
    # Run comparison for all files
    results = compare_all_files(expected_folder, extracted_folder)
    
    # Optionally save results to a file
    if results:
        output_file = "comparison_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    results = main()
#!/usr/bin/env python3
"""
Dataset splitting utility for Prompt Optimizer MVP.
Splits a collection of labeled JSON files into train/val/test sets with reproducible results.
"""

import json
import random
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple
import argparse


def load_examples_from_folder(source_folder: str) -> List[Tuple[Path, Dict[str, Any]]]:
    """
    Load all JSON examples from a folder.
    
    Args:
        source_folder: Path to folder containing JSON files
        
    Returns:
        List of tuples (file_path, example_data)
    """
    source_path = Path(source_folder)
    if not source_path.exists():
        raise ValueError(f"Source folder does not exist: {source_folder}")
    
    examples = []
    json_files = list(source_path.glob("*.json"))
    
    if not json_files:
        raise ValueError(f"No JSON files found in source folder: {source_folder}")
    
    print(f"Found {len(json_files)} JSON files in {source_folder}")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                example_data = json.load(f)
                
            # Validate basic structure
            if not isinstance(example_data, dict):
                print(f"Warning: {json_file.name} is not a valid JSON object")
                continue
            
            # Check for required fields
            if not all(key in example_data for key in ['file_id', 'raw_text', 'gold']):
                print(f"Warning: {json_file.name} missing required fields")
                continue
                
            examples.append((json_file, example_data))
            
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load {json_file.name}: {e}")
            continue
    
    if not examples:
        raise ValueError(f"No valid examples found in {source_folder}")
    
    print(f"Successfully loaded {len(examples)} valid examples")
    return examples


def split_examples(examples: List[Tuple[Path, Dict[str, Any]]], 
                  train_ratio: float = 0.6,
                  val_ratio: float = 0.3, 
                  test_ratio: float = 0.1,
                  random_seed: int = 42) -> Tuple[List, List, List]:
    """
    Split examples into train/val/test sets.
    
    Args:
        examples: List of (file_path, example_data) tuples
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_examples, val_examples, test_examples)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    total_examples = len(examples)
    
    # Calculate split sizes
    train_size = int(total_examples * train_ratio)
    val_size = int(total_examples * val_ratio)
    test_size = total_examples - train_size - val_size  # Remainder goes to test
    
    print(f"Splitting {total_examples} examples:")
    print(f"  Train: {train_size} ({train_size/total_examples:.1%})")
    print(f"  Val: {val_size} ({val_size/total_examples:.1%})")
    print(f"  Test: {test_size} ({test_size/total_examples:.1%})")
    
    # Shuffle with fixed seed for reproducibility
    random.seed(random_seed)
    shuffled_examples = examples.copy()
    random.shuffle(shuffled_examples)
    
    # Split the data
    train_examples = shuffled_examples[:train_size]
    val_examples = shuffled_examples[train_size:train_size + val_size]
    test_examples = shuffled_examples[train_size + val_size:]
    
    return train_examples, val_examples, test_examples


def copy_examples_to_folder(examples: List[Tuple[Path, Dict[str, Any]]], 
                           destination_folder: str,
                           folder_name: str):
    """
    Copy examples to a destination folder.
    
    Args:
        examples: List of (file_path, example_data) tuples
        destination_folder: Base destination path
        folder_name: Name of the subfolder (train/val/test)
    """
    dest_path = Path(destination_folder) / folder_name
    dest_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Copying {len(examples)} files to {dest_path}")
    
    for file_path, example_data in examples:
        dest_file = dest_path / file_path.name
        
        # Copy the file
        try:
            with open(dest_file, 'w', encoding='utf-8') as f:
                json.dump(example_data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Warning: Failed to copy {file_path.name}: {e}")


def create_sample_data_file(destination_folder: str):
    """
    Create a sample_data.json file showing the expected format.
    
    Args:
        destination_folder: Where to create the sample file
    """
    sample_data = {
        "file_id": "sample_001",
        "raw_text": "=== CUSTOMER INFORMATION ===\nDr.: Sample Doctor\nPatient Name: John Doe\n\n=== SUPPLIER INFORMATION ===\nSample Medical Center\n123 Main St, City - 12345\n\n=== ADDITIONAL DOCUMENT DETAILS ===\n\nDate: 1/1/24 Invoice No: 1001 Total: 150.00\nService: Sample Consultation Amount: 150.00",
        "gold": {
            "invoice_number": "1001",
            "patient_name": "John Doe",
            "services": [
                {
                    "service": "Sample Consultation",
                    "amount": 150.0,
                    "quantity": 1,
                    "department": "consultation",
                    "unit": "session",
                    "mrp": 0,
                    "cgst": 0,
                    "cgst_type": None,
                    "sgst": 0,
                    "sgst_type": None,
                    "gst": 0,
                    "gst_type": None
                }
            ],
            "total_amount": 150.0,
            "doctor_name": "Dr. Sample Doctor",
            "facility": "Sample Medical Center, 123 Main St, City - 12345",
            "invoice_date": "1/1/24",
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
    
    sample_file = Path(destination_folder) / "sample_data.json"
    
    try:
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
        print(f"Created sample data file: {sample_file}")
    except IOError as e:
        print(f"Warning: Failed to create sample data file: {e}")


def main():
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test sets')
    parser.add_argument('source_folder', help='Source folder containing JSON files')
    parser.add_argument('--output', '-o', default='data', 
                       help='Output folder (default: data)')
    parser.add_argument('--train-ratio', type=float, default=0.6,
                       help='Training set ratio (default: 0.6)')
    parser.add_argument('--val-ratio', type=float, default=0.3,
                       help='Validation set ratio (default: 0.3)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='Test set ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--sample-data', action='store_true',
                       help='Create a sample_data.json file')
    
    args = parser.parse_args()
    
    try:
        print("=== Dataset Splitting Utility ===")
        print(f"Source folder: {args.source_folder}")
        print(f"Output folder: {args.output}")
        print(f"Random seed: {args.seed}")
        
        # Load examples
        examples = load_examples_from_folder(args.source_folder)
        
        # Split examples
        train_examples, val_examples, test_examples = split_examples(
            examples, 
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio, 
            test_ratio=args.test_ratio,
            random_seed=args.seed
        )
        
        # Copy to destination folders
        copy_examples_to_folder(train_examples, args.output, 'train')
        copy_examples_to_folder(val_examples, args.output, 'val') 
        copy_examples_to_folder(test_examples, args.output, 'test')
        
        # Create sample data file if requested
        if args.sample_data:
            create_sample_data_file(args.output)
        
        print(f"\n=== Split Complete ===")
        print(f"Training examples: {len(train_examples)}")
        print(f"Validation examples: {len(val_examples)}")
        print(f"Test examples: {len(test_examples)}")
        print(f"Total examples: {len(examples)}")
        
        print(f"\nFiles saved to:")
        print(f"  Train: {Path(args.output) / 'train'}")
        print(f"  Val: {Path(args.output) / 'val'}")
        print(f"  Test: {Path(args.output) / 'test'}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
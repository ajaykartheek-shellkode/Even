#!/usr/bin/env python3
"""
Few-shot selector using sentence transformers for Prompt Optimizer MVP.
Uses k-NN similarity to select relevant examples from the training set.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class FewShotSelector:
    """
    Selects few-shot examples using sentence transformer embeddings.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the few-shot selector.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        print(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.train_embeddings = None
        self.train_examples = None
    
    def load_training_examples(self, train_folder: str) -> List[Dict[str, Any]]:
        """
        Load training examples from a folder.
        
        Args:
            train_folder: Path to folder containing training JSON files
            
        Returns:
            List of training examples
        """
        train_path = Path(train_folder)
        if not train_path.exists():
            raise ValueError(f"Training folder does not exist: {train_folder}")
        
        examples = []
        json_files = list(train_path.glob("*.json"))
        
        if not json_files:
            raise ValueError(f"No JSON files found in training folder: {train_folder}")
        
        print(f"Loading {len(json_files)} training examples from {train_folder}")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    example = json.load(f)
                    
                # Validate example structure
                if not all(key in example for key in ['file_id', 'raw_text', 'gold']):
                    print(f"Warning: Invalid example structure in {json_file}")
                    continue
                    
                examples.append(example)
                
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load {json_file}: {e}")
                continue
        
        if not examples:
            raise ValueError(f"No valid examples loaded from {train_folder}")
        
        print(f"Successfully loaded {len(examples)} training examples")
        return examples
    
    def build_embeddings(self, train_folder: str):
        """
        Build embeddings for all training examples.
        
        Args:
            train_folder: Path to folder containing training JSON files
        """
        # Load training examples
        self.train_examples = self.load_training_examples(train_folder)
        
        # Extract raw text from examples
        texts = [example['raw_text'] for example in self.train_examples]
        
        print("Building embeddings for training examples...")
        
        # Generate embeddings
        self.train_embeddings = self.model.encode(
            texts,
            convert_to_tensor=False,
            show_progress_bar=True,
            batch_size=32
        )
        
        print(f"Built embeddings with shape: {self.train_embeddings.shape}")
    
    def select_examples(self, query_text: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Select k most similar examples for a given query text.
        
        Args:
            query_text: The input text to find similar examples for
            k: Number of examples to select
            
        Returns:
            List of k most similar examples
        """
        if self.train_embeddings is None or self.train_examples is None:
            raise ValueError("Must call build_embeddings() first")
        
        if k <= 0:
            return []
        
        if k > len(self.train_examples):
            k = len(self.train_examples)
        
        # Generate embedding for query text
        query_embedding = self.model.encode([query_text], convert_to_tensor=False)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, self.train_embeddings)[0]
        
        # Get indices of top k most similar examples
        top_indices = np.argsort(similarities)[-k:][::-1]  # Sort in descending order
        
        # Return selected examples with similarity scores
        selected = []
        for idx in top_indices:
            example = self.train_examples[idx].copy()
            example['similarity_score'] = float(similarities[idx])
            selected.append(example)
        
        return selected
    
    def format_examples_for_prompt(self, examples: List[Dict[str, Any]]) -> str:
        """
        Format selected examples for inclusion in a prompt.
        
        Args:
            examples: List of selected examples
            
        Returns:
            Formatted string for prompt inclusion
        """
        if not examples:
            return ""
        
        formatted_examples = []
        
        for i, example in enumerate(examples, 1):
            # Format the gold JSON output
            gold_output = json.dumps({"extracted_invoice_values": example['gold']}, 
                                   indent=2, ensure_ascii=False)
            
            example_str = f"""<EXAMPLE_{i}>
INPUT:
{example['raw_text'][:1000]}{'...' if len(example['raw_text']) > 1000 else ''}

OUTPUT:
{gold_output}
</EXAMPLE_{i}>"""
            
            formatted_examples.append(example_str)
        
        return "\n\n".join(formatted_examples)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about loaded training examples.
        
        Returns:
            Dictionary with statistics
        """
        if self.train_examples is None:
            return {"error": "No training examples loaded"}
        
        stats = {
            "total_examples": len(self.train_examples),
            "avg_text_length": np.mean([len(ex['raw_text']) for ex in self.train_examples]),
            "min_text_length": min([len(ex['raw_text']) for ex in self.train_examples]),
            "max_text_length": max([len(ex['raw_text']) for ex in self.train_examples]),
            "total_services": sum([len(ex['gold'].get('services', [])) for ex in self.train_examples]),
            "avg_services_per_example": np.mean([len(ex['gold'].get('services', [])) for ex in self.train_examples])
        }
        
        return stats


def create_few_shot_prompt_section(selector: FewShotSelector, query_text: str, k: int) -> str:
    """
    Create a few-shot section for inclusion in prompts.
    
    Args:
        selector: Initialized FewShotSelector
        query_text: Query text to find examples for
        k: Number of examples to include
        
    Returns:
        Formatted few-shot section
    """
    if k == 0:
        return ""
    
    selected_examples = selector.select_examples(query_text, k)
    
    if not selected_examples:
        return ""
    
    few_shot_section = f"""
<FEW_SHOT_EXAMPLES>
Here are {len(selected_examples)} similar examples to guide your extraction:

{selector.format_examples_for_prompt(selected_examples)}

Follow the same extraction pattern and JSON structure as shown in these examples.
</FEW_SHOT_EXAMPLES>
"""
    
    return few_shot_section


if __name__ == "__main__":
    # Test the few-shot selector
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python few_shot_selector.py <train_folder>")
        sys.exit(1)
    
    train_folder = sys.argv[1]
    
    try:
        # Initialize selector
        selector = FewShotSelector()
        
        # Build embeddings
        selector.build_embeddings(train_folder)
        
        # Print statistics
        stats = selector.get_statistics()
        print("\n=== Training Set Statistics ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Test with a sample query
        if selector.train_examples:
            test_example = selector.train_examples[0]
            print(f"\n=== Testing with example: {test_example['file_id']} ===")
            
            selected = selector.select_examples(test_example['raw_text'], k=3)
            print(f"Selected {len(selected)} examples:")
            
            for i, ex in enumerate(selected):
                print(f"  {i+1}. {ex['file_id']} (similarity: {ex['similarity_score']:.4f})")
            
            # Show formatted prompt section
            print("\n=== Formatted Few-Shot Section ===")
            few_shot_section = create_few_shot_prompt_section(selector, test_example['raw_text'], 2)
            print(few_shot_section[:500] + "..." if len(few_shot_section) > 500 else few_shot_section)
        
    except Exception as e:
        print(f"Error testing few-shot selector: {e}")
        sys.exit(1)
#!/usr/bin/env python3
"""
Advanced Prompt Optimizer with AI-Generated Prompt Variations.
Combines configuration optimization with AI-generated prompt text optimization.
"""

import json
import argparse
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Import our modules
from bedrock_helper import BedrockHelper, extract_json_from_output
from few_shot_selector import FewShotSelector, create_few_shot_prompt_section  
from evaluator import score_prediction
from prompt_generator import PromptGenerator, create_default_variation_strategies, create_default_prompt_components
from progress_tracker import start_tracking, update_trial_progress, update_phase, complete_tracking, fail_tracking, clear_tracking


class AdvancedPromptOptimizer:
    """
    Advanced optimizer that combines configuration optimization with AI-generated prompt variations.
    """
    
    def __init__(self, region_name: str = 'ap-south-1'):
        self.region_name = region_name
        self.bedrock_helper = BedrockHelper(region_name=region_name)
        self.prompt_generator = PromptGenerator(region_name=region_name)
        self.few_shot_selector = None
        self.train_examples = []
        self.val_examples = []
        
        # Will be populated with generated prompts
        self.instruction_variants = []
        self.variant_metadata = []
        
        # Configuration options (focused)
        self.token_options = [1000, 2000, 3000, 5000, 7000]
        self.temperature_options = [0.0, 0.05, 0.1]  
        self.k_options = [1, 2, 3]
    
    def _get_schema_definition(self) -> str:
        """Get the JSON schema definition."""
        return """{
  "extracted_invoice_values": {
    "invoice_number": "",
    "patient_name": "",
    "services": [
      {
        "service": "",
        "amount": "",
        "quantity": "",
        "department": "",
        "unit": "",
        "mrp": "",
        "cgst": "",
        "cgst_type": "",
        "sgst": "",
        "sgst_type": "",
        "gst": "",
        "gst_type": ""
      }
    ],
    "total_amount": "",
    "doctor_name": "",
    "facility": "",
    "invoice_date": "",
    "payment_mode": "",
    "patient_age": "",
    "patient_gender": "",
    "patient_contact": "",
    "cgst": "",
    "cgst_type": "",
    "sgst": "",
    "sgst_type": "",
    "gst": "",
    "gst_type": "",
    "discount": "",
    "mrp": "",
    "round_off": ""
  }
}"""
    
    def generate_instruction_variants(self, base_prompt: str = None, 
                                    generation_method: str = "variations") -> List[str]:
        """
        Generate instruction variants using AI.
        
        Args:
            base_prompt: Starting prompt (if None, creates a basic one)
            generation_method: "variations", "components", or "both"
            
        Returns:
            List of generated instruction variants
        """
        schema = self._get_schema_definition()
        
        # Create base prompt if not provided
        if base_prompt is None:
            base_prompt = f"""<INVOICE_EXTRACTION_SYSTEM>
<ROLE>
You are an expert medical and pharmacy invoice data extraction AI. Your task is to analyze invoice text and extract structured information into the specified JSON format with absolute precision.
</ROLE>

<EXTRACTION_GUIDELINES>
CRITICAL REQUIREMENTS:
1. Extract ALL services/items - Do not skip any service, medication, or billable item
2. Convert all numeric amounts to numbers (not strings)
3. Use null for missing string fields, 0 for missing numeric fields
4. Preserve original date formats exactly as found
5. Extract complete facility information including address and contact details
6. Identify medical departments: "radiology", "pharmacy", "consultation", "laboratory"
</EXTRACTION_GUIDELINES>

<SCHEMA_DEFINITION>
{schema}
</SCHEMA_DEFINITION>

<OUTPUT_REQUIREMENTS>
Output ONLY valid JSON matching the schema exactly.
Do NOT include explanations, comments, or markdown formatting.
Ensure all numeric values are numbers, not strings.
</OUTPUT_REQUIREMENTS>
</INVOICE_EXTRACTION_SYSTEM>"""
        
        generated_variants = []
        metadata = []
        
        if generation_method in ["variations", "both"]:
            print("🤖 Generating AI-powered prompt variations...")
            
            # Generate variations using different strategies
            strategies = create_default_variation_strategies()[:6]  # Limit to 6 strategies
            
            variations = self.prompt_generator.generate_prompt_variations(
                base_prompt, strategies, schema
            )
            
            for var in variations:
                generated_variants.append(var['prompt_text'])
                metadata.append({
                    'type': 'ai_variation',
                    'strategy': var['strategy'],
                    'variation_id': var['variation_id']
                })
        
        if generation_method in ["components", "both"]:
            print("🧩 Generating component-based prompt combinations...")
            
            # Generate component-based prompts
            components = create_default_prompt_components()
            component_prompts = self.prompt_generator.create_component_based_prompts(
                components, schema
            )
            
            # Limit to top combinations to avoid too many
            for prompt_data in component_prompts[:8]:  # Limit to 8 combinations
                generated_variants.append(prompt_data['prompt_text'])
                metadata.append({
                    'type': 'component_based',
                    'components': prompt_data['components'],
                    'combination_id': prompt_data['combination_id']
                })
        
        # Always include the base prompt as option 0
        final_variants = [base_prompt] + generated_variants
        final_metadata = [{'type': 'base_prompt', 'description': 'Original base prompt'}] + metadata
        
        print(f"✅ Generated {len(final_variants)} total instruction variants:")
        print(f"   - 1 base prompt")
        print(f"   - {len([m for m in final_metadata if m['type'] == 'ai_variation'])} AI variations")
        print(f"   - {len([m for m in final_metadata if m['type'] == 'component_based'])} component combinations")
        
        return final_variants, final_metadata
    
    def load_examples(self, train_folder: str, val_folder: str):
        """Load training and validation examples."""
        print(f"Loading examples from {train_folder} and {val_folder}")
        
        # Load training examples
        train_path = Path(train_folder)
        self.train_examples = []
        for json_file in train_path.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    example = json.load(f)
                    if all(key in example for key in ['file_id', 'raw_text', 'gold']):
                        self.train_examples.append(example)
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")
                continue
        
        # Load validation examples
        val_path = Path(val_folder)
        self.val_examples = []
        for json_file in val_path.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    example = json.load(f)
                    if all(key in example for key in ['file_id', 'raw_text', 'gold']):
                        self.val_examples.append(example)
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")
                continue
        
        print(f"Loaded {len(self.train_examples)} training examples")
        print(f"Loaded {len(self.val_examples)} validation examples")
        
        if not self.train_examples or not self.val_examples:
            raise ValueError("Must have both training and validation examples")
        
        # Initialize few-shot selector
        self.few_shot_selector = FewShotSelector()
        self.few_shot_selector.train_examples = self.train_examples
        
        # Build embeddings for few-shot selection
        print("Building embeddings for few-shot selection...")
        texts = [example['raw_text'] for example in self.train_examples]
        self.few_shot_selector.train_embeddings = self.few_shot_selector.model.encode(
            texts, convert_to_tensor=False, show_progress_bar=True, batch_size=32
        )
    
    def initialize_prompts(self, base_prompt: str = None, generation_method: str = "both"):
        """Initialize prompt variants using AI generation."""
        print("\n🚀 Initializing AI-Generated Prompt Variants...")
        
        self.instruction_variants, self.variant_metadata = self.generate_instruction_variants(
            base_prompt, generation_method
        )
        
        total_combinations = (len(self.instruction_variants) * 
                            len(self.k_options) * 
                            len(self.temperature_options) * 
                            len(self.token_options))
        
        print(f"\n📊 Advanced Search Space:")
        print(f"   Instruction variants: {len(self.instruction_variants)}")
        print(f"   Few-shot examples (k): {self.k_options}")
        print(f"   Temperature options: {self.temperature_options}")
        print(f"   Token limit options: {self.token_options}")
        print(f"   Total combinations: {total_combinations}")
    
    def build_prompt(self, instruction_idx: int, k: int, raw_text: str) -> str:
        """Build a complete prompt with instruction and few-shot examples."""
        
        # Get instruction variant
        instruction = self.instruction_variants[instruction_idx]
        
        # Add few-shot examples
        few_shot_section = ""
        if k > 0 and self.few_shot_selector:
            few_shot_section = create_few_shot_prompt_section(
                self.few_shot_selector, raw_text, k
            )
        
        # Build complete prompt
        prompt = f"""{instruction}

{few_shot_section}

<INVOICE_TEXT_TO_ANALYZE>
{raw_text}
</INVOICE_TEXT_TO_ANALYZE>

Extract the structured data as JSON:"""
        
        return prompt
    
    def evaluate_candidate(self, params: Dict[str, Any]) -> float:
        """Evaluate a candidate prompt configuration."""
        
        instruction_idx = params['instruction_idx']
        k = params['k']
        temperature = params['temperature']
        max_tokens = params['max_tokens']
        model_id = params.get('model_id', 'apac.anthropic.claude-3-7-sonnet-20250219-v1:0')
        
        scores = []
        
        # Get variant metadata for logging
        variant_info = self.variant_metadata[instruction_idx]
        variant_type = variant_info['type']
        
        print(f"\nEvaluating: variant={instruction_idx}({variant_type}), k={k}, temp={temperature}, tokens={max_tokens}")
        
        for i, example in enumerate(self.val_examples):
            try:
                # Build prompt
                prompt = self.build_prompt(instruction_idx, k, example['raw_text'])
                
                # Call Bedrock
                result = self.bedrock_helper.call_bedrock(
                    prompt=prompt,
                    model_id=model_id, 
                    temperature=temperature,
                    max_tokens=max_tokens,
                    use_cache=True
                )
                
                # Extract JSON
                predicted_json = extract_json_from_output(result['output'])
                
                # Score prediction
                score = score_prediction(predicted_json, example['gold'])
                scores.append(score['composite'])
                
                print(f"  Example {i+1}/{len(self.val_examples)}: {score['composite']:.3f}")
                
            except Exception as e:
                print(f"  Example {i+1}/{len(self.val_examples)}: ERROR - {e}")
                scores.append(0.0)  # Penalty for errors
        
        mean_score = sum(scores) / len(scores) if scores else 0.0
        print(f"  Mean composite score: {mean_score:.3f}")
        
        return mean_score
    
    def objective(self, trial) -> float:
        """Advanced Optuna objective function with progress tracking."""
        
        # Update trial counter and progress
        self.current_trial = getattr(self, 'current_trial', 0) + 1
        
        # Sample hyperparameters from focused ranges
        instruction_idx = trial.suggest_int('instruction_idx', 0, len(self.instruction_variants) - 1)
        k = trial.suggest_categorical('k', self.k_options)
        temperature = trial.suggest_categorical('temperature', self.temperature_options)  
        max_tokens = trial.suggest_categorical('max_tokens', self.token_options)
        
        params = {
            'instruction_idx': instruction_idx,
            'k': k,
            'temperature': temperature,
            'max_tokens': max_tokens
        }
        
        try:
            # Update progress with current trial info
            variant_type = self.variant_metadata[instruction_idx]['type']
            phase_desc = f"Trial {self.current_trial}: Testing {variant_type} with {k} examples (temp={temperature})"
            update_trial_progress(self.current_trial, 0.0, phase_desc)
            
            score = self.evaluate_candidate(params)
            
            # Update progress with final score
            update_trial_progress(self.current_trial, score, f"Completed trial {self.current_trial} - Score: {score:.3f}")
            
            # Optuna minimizes, so return negative score 
            return 1.0 - score
            
        except Exception as e:
            print(f"Trial failed: {e}")
            update_trial_progress(self.current_trial, 0.0, f"Trial {self.current_trial} failed: {str(e)}")
            return 1.0  # Worst possible score
    
    def optimize(self, n_trials: int = 30) -> Dict[str, Any]:
        """Run advanced optimization with progress tracking."""
        
        print(f"\n🚀 Starting Advanced Prompt Optimization ===")
        print(f"Trials: {n_trials}")
        print(f"Validation examples: {len(self.val_examples)}")
        print(f"Instruction variants: {len(self.instruction_variants)}")
        
        # Start progress tracking
        try:
            start_tracking(n_trials, "Initializing advanced prompt optimization")
            update_phase("Setting up Optuna study and generating variants")
        except Exception as e:
            print(f"Warning: Failed to start progress tracking: {e}")
        
        # Create Optuna study
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        )
        
        # Track current trial for progress updates
        self.current_trial = 0
        
        # Run optimization
        start_time = time.time()
        try:
            update_phase("Running Optuna optimization trials")
            study.optimize(self.objective, n_trials=n_trials)
            optimization_time = time.time() - start_time
            
            # Complete progress tracking
            final_score = 1.0 - study.best_value
            complete_tracking(final_score)
        except Exception as e:
            fail_tracking(str(e))
            raise
        
        print(f"\n🎉 Advanced Optimization Complete ===")
        print(f"Time taken: {optimization_time:.2f} seconds")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best score: {1.0 - study.best_value:.3f}")
        
        best_params = study.best_params
        print(f"Best parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        
        # Get winning variant metadata
        winning_variant = self.variant_metadata[best_params['instruction_idx']]
        print(f"\nWinning prompt variant:")
        print(f"  Type: {winning_variant['type']}")
        if 'strategy' in winning_variant:
            print(f"  Strategy: {winning_variant['strategy']}")
        
        # Generate best prompt example
        if self.val_examples:
            best_prompt = self.build_prompt(
                best_params['instruction_idx'],
                best_params['k'],
                self.val_examples[0]['raw_text'][:500] + "..."
            )
        else:
            best_prompt = "No validation examples available"
        
        return {
            'best_params': best_params,
            'best_score': 1.0 - study.best_value,
            'optimization_time': optimization_time,
            'n_trials': n_trials,
            'best_prompt_example': best_prompt,
            'winning_variant_metadata': winning_variant,
            'all_variant_metadata': self.variant_metadata,
            'study_stats': {
                'n_trials': len(study.trials),
                'best_trial': study.best_trial.number,
                'optimization_time': optimization_time
            }
        }
    
    def save_best_prompt(self, optimization_result: Dict[str, Any], output_file: str = "best_prompt_advanced.json"):
        """Save the advanced optimization results."""
        
        best_params = optimization_result['best_params']
        
        # Get the winning instruction text
        instruction_text = self.instruction_variants[best_params['instruction_idx']]
        
        best_config = {
            'optimization_results': optimization_result,
            'prompt_configuration': {
                'instruction_variant': best_params['instruction_idx'],
                'few_shot_k': best_params['k'],
                'temperature': best_params['temperature'],
                'max_tokens': best_params['max_tokens'],
                'model_id': 'apac.anthropic.claude-3-7-sonnet-20250219-v1:0',
                'use_few_shot': True
            },
            'instruction_text': instruction_text,
            'winning_variant_metadata': optimization_result['winning_variant_metadata'],
            'ai_generation_info': {
                'total_variants_generated': len(self.instruction_variants),
                'variant_types': [meta['type'] for meta in self.variant_metadata],
                'generation_method': 'AI-powered prompt optimization'
            },
            'usage_instructions': {
                'description': 'Advanced optimization with AI-generated prompt variations',
                'innovation': 'Combines configuration optimization with AI prompt text generation',
                'steps': [
                    'AI generated multiple prompt variations using different strategies',
                    'Optimizer tested combinations of prompts + configurations',
                    'Best performing combination selected based on validation data',
                    'Use the winning instruction_text with specified parameters'
                ]
            },
            'metadata': {
                'created_at': time.time(),
                'optimization_score': optimization_result['best_score'],
                'region': self.region_name,
                'training_examples': len(self.train_examples),
                'validation_examples': len(self.val_examples),
                'optimization_type': 'advanced_ai_generated'
            }
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(best_config, f, indent=2, ensure_ascii=False)
            print(f"Advanced optimization results saved to: {output_file}")
        except IOError as e:
            print(f"Error saving results: {e}")


def main():
    parser = argparse.ArgumentParser(description='Advanced prompt optimization with AI-generated variations')
    parser.add_argument('--train-folder', required=True, help='Path to training data folder')
    parser.add_argument('--val-folder', required=True, help='Path to validation data folder')
    parser.add_argument('--trials', type=int, default=50, help='Number of optimization trials')
    parser.add_argument('--model-id', default='apac.anthropic.claude-3-7-sonnet-20250219-v1:0',
                       help='Bedrock model ID')
    parser.add_argument('--region', default='ap-south-1', help='AWS region')
    parser.add_argument('--output', default='best_prompt_advanced.json', help='Output file')
    parser.add_argument('--generation-method', choices=['variations', 'components', 'both'], 
                       default='both', help='Prompt generation method')
    
    args = parser.parse_args()
    
    try:
        print("🤖 Advanced Prompt Optimizer with AI Generation 🤖")
        print("=" * 55)
        print(f"Training folder: {args.train_folder}")
        print(f"Validation folder: {args.val_folder}")
        print(f"Model ID: {args.model_id}")
        print(f"Region: {args.region}")
        print(f"Trials: {args.trials}")
        print(f"Generation method: {args.generation_method}")
        
        # Create optimizer
        optimizer = AdvancedPromptOptimizer(region_name=args.region)
        
        # Load examples
        optimizer.load_examples(args.train_folder, args.val_folder)
        
        # Generate prompt variants
        optimizer.initialize_prompts(generation_method=args.generation_method)
        
        # Run optimization
        result = optimizer.optimize(n_trials=args.trials)
        
        # Save results
        optimizer.save_best_prompt(result, args.output)
        
        print(f"\n🎉 Advanced Optimization Complete!")
        print(f"Best score: {result['best_score']:.3f}")
        print(f"Winning variant type: {result['winning_variant_metadata']['type']}")
        print(f"Results saved to: {args.output}")
        print(f"Cache directory: cache/bedrock/")
        
        # Clear progress tracking after saving results
        print("🧹 Clearing progress tracking...")
        clear_tracking()
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
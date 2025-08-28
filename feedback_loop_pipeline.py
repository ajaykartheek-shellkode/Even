"""
Feedback Loop Pipeline - Based on direct_mipro_test.py
Adapted for input_data and expected_output files with extraction logic
"""

import json
import boto3
import botocore
from pathlib import Path
from deepDiffCode import validate_json
from extraction_code.invoices_folder_automated import extract_invoice

class FeedbackLoopOptimizer:
    """Feedback-driven prompt optimization pipeline"""
    
    def __init__(self, region_name='ap-south-1'):
        config = botocore.config.Config(
            read_timeout=1800,
            connect_timeout=1800,
            retries={'max_attempts': 10}
        )
        self.client = boto3.client('bedrock-runtime', region_name=region_name, config=config)
        self.optimization_history = []
        self.total_api_calls = 0
    
    def call_bedrock(self, prompt):
        """Make Bedrock API call with rate limiting"""
        try:
            self.total_api_calls += 1
            
            response = self.client.converse(
                modelId="apac.anthropic.claude-3-7-sonnet-20250219-v1:0",
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"temperature": 0.0, "maxTokens": 8000}
            )
            result = response['output']['message']['content'][0]['text']
            return result
        except Exception as e:
            print(f"Bedrock API error: {e}")
            return None
    
    def create_base_prompt(self, raw_text):
        """Create base extraction prompt"""
        return f"""Extract structured invoice data from the following raw text and return ONLY valid JSON:

Raw text:
{raw_text}

Return JSON in this exact format:
{{
  "extracted_invoice_values": {{
    "invoice_number": "",
    "patient_name": "",
    "services": [{{
      "service": "",
      "amount": 0,
      "quantity": 0,
      "department": "",
      "unit": null,
      "mrp": 0,
      "cgst": 0,
      "cgst_type": null,
      "sgst": 0,
      "sgst_type": null,
      "gst": 0,
      "gst_type": null
    }}],
    "total_amount": 0,
    "doctor_name": "",
    "facility": "",
    "invoice_date": "",
    "payment_mode": null,
    "patient_age": "",
    "patient_gender": "",
    "patient_contact": null,
    "cgst": 0,
    "cgst_type": null,
    "sgst": 0,
    "sgst_type": null,
    "gst": 0,
    "gst_type": null,
    "discount": 0,
    "mrp": 0,
    "round_off": 0
  }}
}}

Extract all available information accurately from the text.
Return only the JSON, no other text."""
    
    def create_optimized_prompt_with_llm(self, current_prompt, raw_text, differences, expected_json, actual_json):
        """Use LLM to create optimized prompt based on feedback"""
        
        optimization_request = f"""You are a prompt optimization expert. Analyze the extraction errors and create an improved prompt.

CURRENT PROMPT:
{current_prompt}

EXPECTED OUTPUT:
{json.dumps(expected_json, indent=2)}

ACTUAL OUTPUT:
{json.dumps(actual_json, indent=2) if actual_json else "null"}

EXTRACTION ERRORS FOUND:
{differences}

SAMPLE RAW TEXT:
{raw_text[:800]}...

Your task: Create an IMPROVED extraction prompt that fixes these specific errors.

Requirements:
1. Address the specific differences found
2. Add precise instructions to fix extraction issues
3. Keep the same JSON output format
4. Make instructions clear and actionable

Return ONLY the improved prompt text, nothing else."""

        print("Generating LLM-optimized prompt...")
        optimized_prompt = self.call_bedrock(optimization_request)
        
        if optimized_prompt:
            # Clean up the response to extract just the prompt
            lines = optimized_prompt.strip().split('\n')
            # Remove any meta-commentary and return clean prompt
            cleaned_lines = []
            for line in lines:
                if not (line.startswith('Here') or line.startswith('This improved') or 
                       line.startswith('The new prompt') or line.startswith('I have')):
                    cleaned_lines.append(line)
            
            return '\n'.join(cleaned_lines).strip()
        
        # Fallback to current prompt if LLM fails
        print("LLM optimization failed, using current prompt")
        return current_prompt
    
    def extract_with_prompt(self, raw_text, prompt):
        """Extract using custom prompt"""
        response = self.call_bedrock(prompt)
        if not response:
            return None
        
        # Parse JSON from response
        try:
            return json.loads(response)
        except:
            # Try to find JSON in response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
        return None
    
    def optimize_single_file(self, input_file, expected_file, max_iterations=3):
        """Run feedback optimization for single file"""
        
        print(f"Processing: {Path(input_file).name}")
        print("="*80)
        
        # Get raw text from extraction
        extraction_result = extract_invoice(str(input_file))
        raw_text = extraction_result.get('raw_text', '')
        
        if not raw_text:
            print("No raw text extracted from image")
            return None
        
        # Load expected output
        with open(expected_file, 'r') as f:
            expected = json.load(f)
        
        best_score = 0.0
        best_result = None
        best_prompt = None
        
        # Start with base prompt
        current_prompt = self.create_base_prompt(raw_text)
        
        for iteration in range(max_iterations):
            print(f"\n--- ITERATION {iteration + 1} ---")
            
            # Extract with current prompt
            print(f"Extracting with prompt {iteration + 1}...")
            extracted = self.extract_with_prompt(raw_text, current_prompt)
            
            if not extracted:
                print("Extraction failed")
                continue
            
            # Validate
            validation = validate_json(expected, extracted)
            score = validation["score"]
            
            print(f"Score: {score:.3f}")
            if score < 1.0:
                print(f"Issues: {validation['differences_pretty'][:150]}...")
            
            # Store results
            self.optimization_history.append({
                "file": Path(input_file).name,
                "iteration": iteration + 1,
                "score": score,
                "prompt": current_prompt,
                "extracted": extracted,
                "differences": validation["differences_pretty"]
            })
            
            # Track best
            if score > best_score:
                best_score = score
                best_result = extracted
                best_prompt = current_prompt
                print(f"New best score: {best_score:.3f}")
            
            # Check if target achieved
            if score >= 0.99:
                print("Target score achieved!")
                break
            
            # Create optimized prompt for next iteration using LLM
            if iteration < max_iterations - 1:
                print("Creating LLM-optimized prompt...")
                current_prompt = self.create_optimized_prompt_with_llm(
                    current_prompt,
                    raw_text, 
                    validation["differences_pretty"],
                    expected,
                    extracted
                )
        
        return {
            "file": Path(input_file).name,
            "best_score": best_score,
            "best_result": best_result,
            "best_prompt": best_prompt,
            "iterations": len([h for h in self.optimization_history if h["file"] == Path(input_file).name])
        }
    
    def run_feedback_pipeline(self):
        """Run complete feedback pipeline on all files"""
        
        print("FEEDBACK LOOP OPTIMIZATION PIPELINE")
        print("="*80)
        
        # Find all test files
        input_dir = Path("input_data")
        expected_dir = Path("expected_output")
        
        test_files = []
        for input_file in input_dir.glob("*.jpg"):
            expected_file = expected_dir / f"{input_file.stem}_output.json"
            if expected_file.exists():
                test_files.append((input_file, expected_file))
        
        print(f"Found {len(test_files)} test files to process")
        
        results = []
        
        for input_file, expected_file in test_files:
            try:
                result = self.optimize_single_file(input_file, expected_file)
                if result:
                    results.append(result)
                    
                print(f"\nCompleted: {result['file']} - Score: {result['best_score']:.3f}")
                
            except Exception as e:
                print(f"Error processing {input_file}: {e}")
        
        # Show final results
        self.show_pipeline_results(results)
        
        return results
    
    def show_pipeline_results(self, results):
        """Show complete pipeline results"""
        
        print(f"\n{'='*80}")
        print("FEEDBACK PIPELINE RESULTS")
        print(f"{'='*80}")
        
        for result in results:
            print(f"\nFile: {result['file']}")
            print(f"Best Score: {result['best_score']:.3f}")
            print(f"Iterations: {result['iterations']}")
        
        if results:
            avg_score = sum(r['best_score'] for r in results) / len(results)
            success_count = sum(1 for r in results if r['best_score'] >= 0.95)
            
            print(f"\nSUMMARY:")
            print(f"Average Score: {avg_score:.3f}")
            print(f"Success Rate: {success_count}/{len(results)} ({success_count/len(results):.1%})")
            print(f"Total API Calls: {self.total_api_calls}")
        
        print(f"\n{'='*80}")
        print("PROMPT EVOLUTION EXAMPLES")
        print(f"{'='*80}")
        
        # Show prompt evolution for first file
        if self.optimization_history:
            first_file = self.optimization_history[0]["file"]
            file_history = [h for h in self.optimization_history if h["file"] == first_file]
            
            for entry in file_history:
                print(f"\nIteration {entry['iteration']} (Score: {entry['score']:.3f}):")
                print("-" * 40)
                prompt_preview = entry['prompt'][:200] + "..." if len(entry['prompt']) > 200 else entry['prompt']
                print(prompt_preview)

def main():
    """Run the feedback loop pipeline"""
    
    optimizer = FeedbackLoopOptimizer()
    results = optimizer.run_feedback_pipeline()
    
    if results:
        print(f"\nPIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Processed {len(results)} files with feedback optimization")
    else:
        print("No files processed successfully")

if __name__ == "__main__":
    main()
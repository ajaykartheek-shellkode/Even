import json
import boto3
import os
from pathlib import Path


class SimplePromptOptimizer:
    def __init__(self):
        # AWS Bedrock setup
        self.bedrock = boto3.client('bedrock-runtime', region_name='ap-south-1')
        
        # Get base prompt from your existing extraction code
        self.base_prompt = self.get_base_prompt()
        
        # Schema from your extraction code
        self.schema = self.get_schema()
        
        self.api_calls = 0
    
    def get_base_prompt(self):
        """Extract the base prompt from your existing extraction code"""
        extraction_file = "extraction-code/invoices_folder_automated.py"
        
        with open(extraction_file, 'r') as f:
            content = f.read()
        
        # Find the prompt section
        start_marker = 'prompt = f"""'
        end_marker = '"""'
        
        start_idx = content.find(start_marker)
        if start_idx == -1:
            raise Exception("Base prompt not found in extraction code")
        
        start_idx += len(start_marker)
        end_idx = content.find(end_marker, start_idx)
        
        prompt = content[start_idx:end_idx].strip()
        return prompt
    
    def get_schema(self):
        """Get schema from your deepDiffCode.py"""
        # Read one of your expected output files to get the schema
        expected_files = list(Path("expected-output").glob("*.json"))
        if not expected_files:
            raise Exception("No expected output files found")
        
        with open(expected_files[0], 'r') as f:
            data = json.load(f)
        
        return json.dumps(data.get('extracted_invoice_values', {}), indent=2)
    
    def call_bedrock(self, prompt):
        """Simple Bedrock API call"""
        try:
            response = self.bedrock.converse(
                modelId="apac.anthropic.claude-3-7-sonnet-20250219-v1:0",
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"temperature": 0.0, "maxTokens": 10000}
            )
            self.api_calls += 1
            return response['output']['message']['content'][0]['text']
        except Exception as e:
            print(f"API Error: {e}")
            return None
    
    def run_deepdiff_comparison(self):
        """Use your existing deepdiff logic"""
        from deepDiffCode import compare_all_files
        return compare_all_files("expected-output", "extracted-output")
    
    def optimize_prompt(self, feedback):
        """Simple prompt optimization using feedback"""
        optimization_prompt = f"""
        The current invoice extraction prompt is not performing well. Here's the feedback:
        
        FEEDBACK: {feedback}
        
        CURRENT PROMPT:
        {self.base_prompt}
        
        Please improve this prompt to fix the issues mentioned in the feedback. 
        Return only the improved prompt, nothing else.
        """
        
        improved_prompt = self.call_bedrock(optimization_prompt)
        if improved_prompt:
            self.base_prompt = improved_prompt
            print("✅ Prompt optimized based on feedback")
        else:
            print("❌ Failed to optimize prompt")
    
    def run_pipeline(self):
        """Main optimization pipeline"""
        print("🚀 Starting Simple Prompt Optimization Pipeline")
        print("=" * 50)
        
        iteration = 1
        max_iterations = 3
        
        while iteration <= max_iterations:
            print(f"\n📍 Iteration {iteration}")
            print("-" * 30)
            
            # Step 1: Test current prompt with your deepdiff logic
            print("1. Running deepdiff comparison...")
            comparison_results = self.run_deepdiff_comparison()
            
            # Step 2: Check if we need optimization  
            has_mismatches = False
            if comparison_results:
                for result in comparison_results:
                    if result.get('valid') == False or result.get('differences'):
                        has_mismatches = True
                        break
            
            if has_mismatches:
                print("2. Mismatches found - optimizing prompt...")
                
                # Step 3: Get feedback and optimize
                feedback = f"Comparison results: {comparison_results}"
                self.optimize_prompt(feedback)
                
            else:
                print("2. ✅ No major issues found!")
                break
            
            iteration += 1
        
        print(f"\n🎯 Optimization complete!")
        print(f"Total API calls: {self.api_calls}")
        
        # Save optimized prompt
        with open("optimized_prompt.txt", "w") as f:
            f.write(self.base_prompt)
        
        print("💾 Optimized prompt saved to optimized_prompt.txt")


if __name__ == "__main__":
    optimizer = SimplePromptOptimizer()
    optimizer.run_pipeline()
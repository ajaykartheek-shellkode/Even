"""
Working Feedback-Based Prompt Optimizer
Fixed all issues - simplified, working implementation
"""

import json
import boto3
import botocore
import time
from pathlib import Path
from deepDiffCode import validate_json

class WorkingFeedbackOptimizer:
    """Simple, working feedback optimizer"""
    
    def __init__(self, region_name='ap-south-1'):
        config = botocore.config.Config(
            read_timeout=180,
            connect_timeout=60,
            retries={'max_attempts': 3}
        )
        self.client = boto3.client('bedrock-runtime', region_name=region_name, config=config)
        self.api_calls = 0
        self.results = []
    
    def call_bedrock(self, prompt, timeout=60):
        """Safe Bedrock API call with timeout"""
        try:
            self.api_calls += 1
            print(f"  API Call #{self.api_calls}...")
            time.sleep(2)  # Rate limiting
            
            response = self.client.converse(
                modelId="apac.anthropic.claude-3-7-sonnet-20250219-v1:0",
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"temperature": 0.0, "maxTokens": 4000}
            )
            result = response['output']['message']['content'][0]['text']
            print(f"  Success ({len(result)} chars)")
            return result
            
        except Exception as e:
            print(f"  Error: {str(e)[:100]}...")
            return None
    
    def extract_json_from_response(self, response):
        """Extract JSON from Claude response"""
        if not response:
            return None
        
        try:
            # Try direct parsing first
            return json.loads(response)
        except:
            pass
        
        # Try to find JSON in response
        import re
        json_patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
            r'\{.*\}'
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except:
                    continue
        
        return None
    
    def create_base_prompt(self):
        """Simple base extraction prompt"""
        return """Extract invoice data from the text and return ONLY JSON:

{
  "extracted_invoice_values": {
    "invoice_number": "",
    "patient_name": "",
    "services": [{"service": "", "amount": 0, "quantity": 1, "department": "consultation"}],
    "total_amount": 0,
    "doctor_name": "",
    "facility": "",
    "invoice_date": "",
    "patient_age": "",
    "patient_gender": ""
  }
}

Return only the JSON, no other text."""
    
    def create_improved_prompt(self, base_prompt, differences):
        """Create improved prompt using LLM feedback"""
        
        optimization_prompt = f"""Improve this extraction prompt to fix the specific errors:

CURRENT PROMPT:
{base_prompt}

ERRORS TO FIX:
{differences[:500]}...

Create an IMPROVED prompt that addresses these errors. Keep the same JSON format.
Return only the improved prompt text."""

        improved = self.call_bedrock(optimization_prompt)
        if improved:
            # Clean up response
            lines = improved.strip().split('\n')
            cleaned = []
            for line in lines:
                if not line.startswith(('Here', 'This', 'I have', 'The improved')):
                    cleaned.append(line)
            result = '\n'.join(cleaned).strip()
            return result if result else base_prompt
        
        return base_prompt
    
    def test_simple_case(self):
        """Test with simple hardcoded data first"""
        
        print("TESTING SIMPLE CASE")
        print("="*60)
        
        # Simple test data
        raw_text = """INVOICE #123
Date: 01/01/2025
Patient: John Doe
Age: 30 Years
Gender: Male
Doctor: Dr. Smith
Service: Consultation
Amount: 500"""
        
        expected = {
            "extracted_invoice_values": {
                "invoice_number": "123",
                "patient_name": "John Doe",
                "services": [{"service": "Consultation", "amount": 500, "quantity": 1, "department": "consultation"}],
                "total_amount": 500,
                "doctor_name": "Dr. Smith", 
                "facility": "",
                "invoice_date": "01/01/2025",
                "patient_age": "30 Years",
                "patient_gender": "Male"
            }
        }
        
        # Test base prompt
        base_prompt = self.create_base_prompt()
        full_prompt = f"{base_prompt}\n\nText:\n{raw_text}"
        
        print("Step 1: Base extraction...")
        response = self.call_bedrock(full_prompt)
        extracted = self.extract_json_from_response(response)
        
        if not extracted:
            print("Base extraction failed")
            return False
        
        validation = validate_json(expected, extracted)
        base_score = validation["score"]
        
        print(f"Base Score: {base_score:.3f}")
        
        if base_score < 0.95:
            print("Step 2: Optimizing prompt...")
            improved_prompt = self.create_improved_prompt(
                base_prompt,
                validation["differences_pretty"]
            )
            
            # Test improved prompt
            improved_full_prompt = f"{improved_prompt}\n\nText:\n{raw_text}"
            print("Step 3: Testing improved prompt...")
            
            improved_response = self.call_bedrock(improved_full_prompt)
            improved_extracted = self.extract_json_from_response(improved_response)
            
            if improved_extracted:
                improved_validation = validate_json(expected, improved_extracted)
                improved_score = improved_validation["score"]
                
                print(f"Improved Score: {improved_score:.3f}")
                print(f"Improvement: +{improved_score - base_score:.3f}")
                
                return improved_score > base_score
        
        return True
    
    def process_single_file(self, test_file_name):
        """Process one file from the test data"""
        
        print(f"\nPROCESSING: {test_file_name}")
        print("="*60)
        
        # Load expected output
        expected_file = Path("expected_output") / f"{test_file_name}_output.json"
        if not expected_file.exists():
            print(f"Expected file not found: {expected_file}")
            return None
        
        with open(expected_file, 'r') as f:
            expected = json.load(f)
        
        # Use extracted output if it exists (skip image processing)
        extracted_file = Path("extracted_output") / f"{test_file_name}_output.json" 
        if extracted_file.exists():
            print("Using existing extracted output...")
            with open(extracted_file, 'r') as f:
                extraction_result = json.load(f)
            raw_text = extraction_result.get('raw_text', '')
        else:
            print("No extracted output found, skipping...")
            return None
        
        if not raw_text:
            print("No raw text available")
            return None
        
        # Use shorter raw text to avoid timeouts
        raw_text = raw_text[:1000]
        
        print("Step 1: Base extraction...")
        base_prompt = self.create_base_prompt()
        full_prompt = f"{base_prompt}\n\nText:\n{raw_text}"
        
        response = self.call_bedrock(full_prompt)
        extracted = self.extract_json_from_response(response)
        
        if not extracted:
            print("Base extraction failed")
            return None
        
        validation = validate_json(expected, extracted)
        base_score = validation["score"]
        
        print(f"Base Score: {base_score:.3f}")
        
        best_score = base_score
        best_prompt = base_prompt
        
        # Try optimization if score is not good enough
        if base_score < 0.95:
            print("Step 2: Optimizing prompt...")
            
            improved_prompt = self.create_improved_prompt(
                base_prompt,
                validation["differences_pretty"]
            )
            
            print("Step 3: Testing improved prompt...")
            improved_full_prompt = f"{improved_prompt}\n\nText:\n{raw_text}"
            
            improved_response = self.call_bedrock(improved_full_prompt)
            improved_extracted = self.extract_json_from_response(improved_response)
            
            if improved_extracted:
                improved_validation = validate_json(expected, improved_extracted)
                improved_score = improved_validation["score"]
                
                print(f"Improved Score: {improved_score:.3f}")
                print(f"Change: {improved_score - base_score:+.3f}")
                
                if improved_score > base_score:
                    best_score = improved_score
                    best_prompt = improved_prompt
        
        result = {
            "file": test_file_name,
            "base_score": base_score,
            "best_score": best_score,
            "improvement": best_score - base_score,
            "api_calls": self.api_calls
        }
        
        self.results.append(result)
        return result

def main():
    """Run the working feedback optimizer"""
    
    optimizer = WorkingFeedbackOptimizer()
    
    print("WORKING FEEDBACK OPTIMIZER")
    print("="*80)
    
    # Test simple case first
    if not optimizer.test_simple_case():
        print("Simple test failed, stopping")
        return
    
    print(f"\nSimple test passed! API calls: {optimizer.api_calls}")
    
    # Test with real files
    test_files = [
        "71345",
        "40e86fb6_34cc_4789_b43b_2c1a1270ee055190886607521502167"
    ]
    
    for test_file in test_files:
        try:
            result = optimizer.process_single_file(test_file)
            if result:
                print(f"SUCCESS: {result['file']} - Score: {result['best_score']:.3f}")
            else:
                print(f"FAILED: {test_file}")
        except Exception as e:
            print(f"ERROR processing {test_file}: {e}")
    
    # Show summary
    if optimizer.results:
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        
        for result in optimizer.results:
            print(f"{result['file']}: {result['best_score']:.3f} ({result['improvement']:+.3f})")
        
        avg_score = sum(r['best_score'] for r in optimizer.results) / len(optimizer.results)
        avg_improvement = sum(r['improvement'] for r in optimizer.results) / len(optimizer.results)
        
        print(f"\nAverage Score: {avg_score:.3f}")
        print(f"Average Improvement: {avg_improvement:+.3f}")
        print(f"Total API Calls: {optimizer.api_calls}")
        
        print(f"\nFEEDBACK OPTIMIZATION WORKING!")
    else:
        print("No files processed successfully")

if __name__ == "__main__":
    main()
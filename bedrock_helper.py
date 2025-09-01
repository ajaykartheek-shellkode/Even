#!/usr/bin/env python3
"""
Bedrock helper with caching and retry logic for Prompt Optimizer MVP.
Provides a wrapper around AWS Bedrock runtime with smart caching and robust error handling.
"""

import boto3
import json
import hashlib
import os
import time
import re
from pathlib import Path
from typing import Dict, Any, Optional, Union
from botocore.exceptions import ClientError
import botocore.config


class BedrockHelper:
    """
    A helper class for calling AWS Bedrock with caching and retry logic.
    """
    
    def __init__(self, region_name: str = 'ap-south-1', cache_dir: str = 'cache/bedrock'):
        self.region_name = region_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure boto3 client with robust settings
        config = botocore.config.Config(
            read_timeout=1800,
            connect_timeout=1800,
            retries={'max_attempts': 10}
        )
        
        self.client = boto3.client('bedrock-runtime', region_name=region_name, config=config)
    
    def _generate_cache_key(self, prompt: str, model_id: str, temperature: float, 
                           max_tokens: int) -> str:
        """Generate a unique cache key based on input parameters."""
        content = f"{prompt}|{model_id}|{temperature}|{max_tokens}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load cached response if it exists."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load cache file {cache_file}: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, output: str, meta: Dict[str, Any]):
        """Save response to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "output": output,
                    "meta": meta
                }, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Warning: Failed to save cache file {cache_file}: {e}")
    
    def _call_bedrock_with_retry(self, params: Dict[str, Any], max_retries: int = 3) -> str:
        """Call Bedrock with exponential backoff retry logic."""
        for attempt in range(max_retries):
            try:
                response = self.client.converse(**params)
                return response['output']['message']['content'][0]['text']
            
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                
                if error_code in ['ThrottlingException', 'ServiceQuotaExceededException']:
                    if attempt < max_retries - 1:
                        wait_time = min(60 * (2 ** attempt), 300)  # Max 5 minutes
                        print(f"Rate limited, retrying in {wait_time} seconds (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                
                print(f"Bedrock API error: {e}")
                raise
            
            except Exception as e:
                print(f"Unexpected error calling Bedrock: {e}")
                if attempt < max_retries - 1:
                    time.sleep(10)  # Short wait for other errors
                    continue
                raise
        
        raise RuntimeError(f"Failed to call Bedrock after {max_retries} attempts")
    
    def call_bedrock(self, prompt: str, 
                    model_id: str = "apac.anthropic.claude-3-7-sonnet-20250219-v1:0",
                    temperature: float = 0.0,
                    max_tokens: int = 10000,
                    use_cache: bool = True) -> Dict[str, Any]:
        """
        Call Bedrock with caching support.
        
        Args:
            prompt: The prompt to send to the model
            model_id: The Bedrock model ID to use
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            use_cache: Whether to use caching
            
        Returns:
            Dict containing 'output' and 'meta' keys
        """
        # Generate cache key
        cache_key = self._generate_cache_key(prompt, model_id, temperature, max_tokens)
        
        # Try to load from cache first
        if use_cache:
            cached_result = self._load_from_cache(cache_key)
            if cached_result:
                print(f"Using cached result for key: {cache_key[:12]}...")
                return cached_result
        
        # Prepare Bedrock parameters
        params = {
            "modelId": model_id,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {
                "temperature": temperature,
                "maxTokens": max_tokens
            }
        }
        
        # Call Bedrock
        print(f"Calling Bedrock model {model_id}...")
        start_time = time.time()
        
        try:
            output = self._call_bedrock_with_retry(params)
            call_duration = time.time() - start_time
            
            # Prepare metadata
            meta = {
                "model_id": model_id,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "call_duration": call_duration,
                "timestamp": time.time(),
                "cache_key": cache_key,
                "prompt_length": len(prompt),
                "output_length": len(output)
            }
            
            result = {
                "output": output,
                "meta": meta
            }
            
            # Save to cache
            if use_cache:
                self._save_to_cache(cache_key, output, meta)
                print(f"Cached result with key: {cache_key[:12]}")
            
            print(f"Bedrock call completed in {call_duration:.2f}s")
            return result
        
        except Exception as e:
            print(f"Error in bedrock call: {e}")
            raise


def extract_json_from_output(output: str) -> Dict[str, Any]:
    """
    Extract JSON from model output with robust parsing.
    
    Args:
        output: Raw text output from the model
        
    Returns:
        Parsed JSON dictionary
    """
    # Clean up the output
    cleaned = output.strip()
    
    # Try direct JSON parsing first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Use regex to find JSON object
    json_match = re.search(r'(\{[\s\S]*\})', cleaned, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # More aggressive extraction - find first { and last }
    start_pos = cleaned.find('{')
    end_pos = cleaned.rfind('}')
    
    if start_pos != -1 and end_pos != -1 and end_pos > start_pos:
        try:
            json_str = cleaned[start_pos:end_pos+1]
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # If all parsing fails, return an error structure
    return {
        "error": "Could not parse JSON from model output",
        "raw_output": output
    }


# Compatibility with existing code
def call_bedrock(prompt: str, 
                model_id: str = "apac.anthropic.claude-3-7-sonnet-20250219-v1:0",
                temperature: float = 0.0,
                max_tokens: int = 10000,
                region: str = 'ap-south-1',
                use_cache: bool = True) -> Dict[str, Any]:
    """
    Standalone function for calling Bedrock (compatibility wrapper).
    
    Args:
        prompt: The prompt to send
        model_id: Bedrock model ID  
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        region: AWS region
        use_cache: Whether to use caching
        
    Returns:
        Dict with 'output' and 'meta' keys
    """
    helper = BedrockHelper(region_name=region)
    return helper.call_bedrock(prompt, model_id, temperature, max_tokens, use_cache)


if __name__ == "__main__":
    # Test the bedrock helper
    test_prompt = "What is 2+2? Answer in JSON format: {\"result\": <answer>}"
    
    try:
        helper = BedrockHelper()
        result = helper.call_bedrock(test_prompt, temperature=0.0)
        
        print("=== Raw Result ===")
        print(json.dumps(result, indent=2))
        
        print("\n=== Extracted JSON ===")
        extracted = extract_json_from_output(result['output'])
        print(json.dumps(extracted, indent=2))
        
    except Exception as e:
        print(f"Error testing bedrock helper: {e}")
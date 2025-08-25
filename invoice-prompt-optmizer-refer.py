import boto3
import botocore
import json
import traceback
from langchain_community.document_loaders import AmazonTextractPDFLoader

# -----------------------
# User / AWS config
# -----------------------
REGION_NAME = 'ap-south-1'          # ap-south-1 (Mumbai) supports Bedrock prompt optimization
S3_BUCKET_NAME = 'even-ocr-poc'
MODEL_ID = "apac.amazon.nova-pro-v1:0"   # your Nova model id
# Path to optional small evaluation dataset (JSONL with {"document_text": "...", "expected": {...}} lines)
EVAL_JSONL_PATH = None  # set to a file path if you want the quick evaluation step

# -----------------------
# AWS client config
# -----------------------
config = botocore.config.Config(
    read_timeout=1800,
    connect_timeout=1800,
    retries={'max_attempts': 10}
)

s3_client = boto3.client('s3', region_name=REGION_NAME)
textract_client = boto3.client('textract', region_name=REGION_NAME)
# The runtime client you were already using for converse/inference:
bedrock_runtime_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=REGION_NAME,
    config=config
)
# The agent/runtime client used for prompt optimization (Agents for Bedrock runtime)
prompt_opt_client = boto3.client('bedrock-agent-runtime', region_name=REGION_NAME, config=config)


# -----------------------
# Helpers: S3 + Textract
# -----------------------
def upload_document_to_s3(file_path):
    try:
        file_name = file_path.split('\\')[-1].split('/')[-1]
        with open(file_path, 'rb') as file:
            s3_client.upload_fileobj(file, S3_BUCKET_NAME, file_name)
        s3_uri = f"s3://{S3_BUCKET_NAME}/{file_name}"
        return s3_uri
    except Exception:
        print("Error uploading to S3:", traceback.format_exc())
        raise


def extract_document_text(s3_uri):
    try:
        textract_features = ["LAYOUT"]
        loader = AmazonTextractPDFLoader(s3_uri, textract_features, client=textract_client)
        docs = loader.load()
        extracted_text = "\n".join(doc.page_content for doc in docs)
        return extracted_text
    except Exception:
        print("Error extracting with Textract:", traceback.format_exc())
        raise


# -----------------------
# Prompt building
# -----------------------
BASE_SCHEMA_PROMPT = """
Below are the JSON schema to be extracted from the following document text. Please extract all the relevant information and return it in the exact JSON format specified below.

Document: {document_text}

Required JSON Schema:
{{
  "extracted_invoice_values": {{
    "invoice_number": "",
    "patient_name": "",
    "services": [
      {{
        "service": "",
        "amount": 0,
        "quantity": 0,
        "department": "",
        "unit": "",
        "mrp": 0,
        "cgst": 0,
        "cgst_type": "",
        "sgst": 0,
        "sgst_type": "",
        "gst": 0,
        "gst_type": ""
      }}
    ],
    "total_amount": 0,
    "doctor_name": "",
    "facility": "",
    "invoice_date": "",
    "payment_mode": "",
    "patient_age": "",
    "patient_gender": "",
    "patient_contact": "",
    "cgst": 0,
    "cgst_type": "",
    "sgst": 0,
    "sgst_type": "",
    "gst": 0,
    "gst_type": "",
    "discount": 0,
    "mrp": 0,
    "round_off": 0
  }}
}}

Please return only the JSON object with the extracted values. If a field is not found in the document, leave it empty or as 0 for numeric fields.
"""


# -----------------------
# Prompt optimizer using Amazon Bedrock's OptimizePrompt API (FIXED)
# -----------------------
def get_input(prompt_text):
    """Helper function to format input for optimize_prompt API"""
    return {
        "textPrompt": {
            "text": prompt_text
        }
    }


def handle_response_stream(response):
    """
    Handle the event stream from optimize_prompt API response.
    Returns the optimized prompt text.
    """
    optimized_prompt = None
    try:
        event_stream = response['optimizedPrompt']
        for event in event_stream:
            if 'optimizedPromptEvent' in event:
                print("========================== OPTIMIZED PROMPT RECEIVED ======================")
                optimized_event = event['optimizedPromptEvent']
                print(f"DEBUG: Optimized event type: {type(optimized_event)}")
                print(f"DEBUG: Optimized event content: {optimized_event}")
                
                # Handle different possible structures of the optimized prompt event
                if isinstance(optimized_event, str):
                    # If it's already a string, use it directly
                    optimized_prompt = optimized_event
                elif isinstance(optimized_event, dict):
                    # Try different possible keys where the text might be stored
                    if 'text' in optimized_event:
                        optimized_prompt = optimized_event['text']
                    elif 'content' in optimized_event:
                        optimized_prompt = optimized_event['content']
                    elif 'prompt' in optimized_event:
                        if isinstance(optimized_event['prompt'], str):
                            optimized_prompt = optimized_event['prompt']
                        elif isinstance(optimized_event['prompt'], dict) and 'text' in optimized_event['prompt']:
                            optimized_prompt = optimized_event['prompt']['text']
                    elif 'optimizedPrompt' in optimized_event:
                        if isinstance(optimized_event['optimizedPrompt'], str):
                            optimized_prompt = optimized_event['optimizedPrompt']
                        elif isinstance(optimized_event['optimizedPrompt'], dict) and 'text' in optimized_event['optimizedPrompt']:
                            optimized_prompt = optimized_event['optimizedPrompt']['text']
                    else:
                        # If none of the expected keys are found, try to extract any string value
                        for key, value in optimized_event.items():
                            if isinstance(value, str) and len(value) > 100:  # Assume the prompt is reasonably long
                                optimized_prompt = value
                                break
                        
                        # If still no text found, convert to JSON string as last resort
                        if not optimized_prompt:
                            print("WARNING: Could not find text in optimized event, using JSON string")
                            optimized_prompt = json.dumps(optimized_event)
                
                print(f"DEBUG: Extracted optimized prompt length: {len(optimized_prompt) if optimized_prompt else 0}")
                break
                
            elif 'analyzePromptEvent' in event:
                print("========================= ANALYZE PROMPT EVENT =======================")
                analyze_prompt = event['analyzePromptEvent']
                print("Analysis:", analyze_prompt)
                
    except Exception as e:
        print(f"Error processing response stream: {e}")
        print(traceback.format_exc())
        raise e
    
    return optimized_prompt


def optimize_prompt_with_bedrock(prompt_text, target_model_id=MODEL_ID):
    """
    Uses Bedrock Agents runtime's optimize_prompt to rewrite a prompt for the target model.
    Returns optimized_text (string). If optimization fails, returns original prompt.
    """
    try:
        print(f"[optimize_prompt] Optimizing prompt for model: {target_model_id}")
        print(f"[optimize_prompt] Original prompt length: {len(prompt_text)}")
        
        response = prompt_opt_client.optimize_prompt(
            input=get_input(prompt_text),
            targetModelId=target_model_id
        )
        
        print("Request ID:", response.get("ResponseMetadata", {}).get("RequestId"))
        print("========================== INPUT PROMPT ======================")
        print(prompt_text[:500] + "..." if len(prompt_text) > 500 else prompt_text)
        
        optimized_text = handle_response_stream(response)
        
        if optimized_text:
            print(f"[optimize_prompt] Successfully optimized prompt (length: {len(optimized_text)})")
            # Verify that the optimized prompt still contains the placeholder
            if '{document_text}' not in optimized_text:
                print("[optimize_prompt] WARNING: Optimized prompt doesn't contain {document_text} placeholder")
                print("[optimize_prompt] Adding placeholder to optimized prompt...")
                # Try to intelligently add the placeholder
                if "Document:" in optimized_text:
                    # Replace any existing document reference with our placeholder
                    import re
                    optimized_text = re.sub(r'Document:.*?(?=Required|JSON|Schema|\n\n)', 'Document: {document_text}\n\n', optimized_text, flags=re.DOTALL)
                else:
                    # Add the document reference if not found
                    optimized_text = f"Document: {{document_text}}\n\n{optimized_text}"
            
            return optimized_text
        else:
            print("[optimize_prompt] No optimized prompt received - returning original prompt")
            return prompt_text
            
    except Exception as e:
        print(f"Prompt optimization failed (falling back to original). Details: {e}")
        print(traceback.format_exc())
        return prompt_text


# -----------------------
# LLM call (converse) - uses bedrock_runtime_client as before
# -----------------------
def llm_call_with_converse(prompt_text, model_id=MODEL_ID, temperature=0.0, max_tokens=4096):
    """
    Sends prompt_text via the converse API to the target model and returns the textual response.
    """
    try:
        inference_config = {"temperature": temperature, "maxTokens": max_tokens}
        converse_api_params = {
            "modelId": model_id,
            "messages": [{"role": "user", "content": [{"text": prompt_text}]}],
            "inferenceConfig": inference_config
        }
        response = bedrock_runtime_client.converse(**converse_api_params)
        
        # Extract text from response
        try:
            result_text = response['output']['message']['content'][0]['text']
            return result_text
        except (KeyError, IndexError) as e:
            # Fallback to stringified response if structure differs
            print(f"Unexpected response structure: {e}")
            return json.dumps(response)
            
    except Exception as e:
        print(f"LLM call failed: {e}")
        print(traceback.format_exc())
        raise


# -----------------------
# Quick evaluation utilities (optional)
# -----------------------
def safe_extract_json_from_text(text):
    """
    Try to extract the first JSON object from an LLM response string.
    Returns parsed object on success, else None.
    """
    try:
        # Try to find JSON object boundaries
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = text[start:end+1]
        return json.loads(candidate)
    except Exception:
        return None


def score_parsed_output(parsed_json, required_keys=None):
    """
    Simple scoring: counts how many required_keys exist non-empty in the parsed_json.
    """
    if not parsed_json or not required_keys:
        return 0
    score = 0
    for k in required_keys:
        parts = k.split('.')
        node = parsed_json
        found = True
        for p in parts:
            if isinstance(node, dict) and p in node:
                node = node[p]
            else:
                found = False
                break
        if found and node not in [None, "", [], {}]:
            score += 1
    return score


def quick_evaluate_prompts(original_prompt_template, optimized_prompt_template, examples, required_keys):
    """
    Run the original and optimized prompts on a small set of examples.
    Returns simple aggregate scores.
    """
    results = {
        "original": {"scores": [], "examples": []},
        "optimized": {"scores": [], "examples": []}
    }

    for ex in examples:
        doc_text = ex.get("document_text", "")
        # Build final prompts by substituting document text
        try:
            prompt_orig = original_prompt_template.format(document_text=doc_text)
            prompt_opt = optimized_prompt_template.format(document_text=doc_text)

            resp_orig_text = llm_call_with_converse(prompt_orig)
            resp_opt_text = llm_call_with_converse(prompt_opt)

            parsed_orig = safe_extract_json_from_text(resp_orig_text)
            parsed_opt = safe_extract_json_from_text(resp_opt_text)

            score_orig = score_parsed_output(parsed_orig, required_keys)
            score_opt = score_parsed_output(parsed_opt, required_keys)

            results["original"]["scores"].append(score_orig)
            results["original"]["examples"].append({"resp": resp_orig_text, "parsed": parsed_orig})
            results["optimized"]["scores"].append(score_opt)
            results["optimized"]["examples"].append({"resp": resp_opt_text, "parsed": parsed_opt})
            
        except Exception as e:
            print(f"Error evaluating example: {e}")
            # Add zero scores for failed examples
            results["original"]["scores"].append(0)
            results["original"]["examples"].append({"resp": "", "parsed": None})
            results["optimized"]["scores"].append(0)
            results["optimized"]["examples"].append({"resp": "", "parsed": None})

    # Calculate aggregates
    def agg(scores):
        return {
            "count": len(scores), 
            "sum": sum(scores), 
            "avg": float(sum(scores))/len(scores) if scores else 0.0
        }

    results["original"]["aggregate"] = agg(results["original"]["scores"])
    results["optimized"]["aggregate"] = agg(results["optimized"]["scores"])

    return results


# -----------------------
# Main orchestrator: process_document
# -----------------------
def process_document(file_path, run_quick_eval=False, eval_examples=None):
    """
    Orchestrates:
    1. Upload -> Textract
    2. Build prompt (base), run Bedrock prompt optimizer to rewrite for MODEL_ID
    3. Optionally run quick evaluation on eval_examples (small list)
    4. Call model with optimized prompt and return result string
    """
    try:
        # Step 1: Upload and extract text
        print("[process_document] Uploading document and extracting text...")
        s3_uri = upload_document_to_s3(file_path)
        document_text = extract_document_text(s3_uri)
        if not document_text:
            raise Exception("No text extracted from the document.")

        print(f"[process_document] Extracted text length: {len(document_text)}")

        # Step 2: Optimize prompt
        print("[process_document] Submitting prompt for optimization...")
        optimized_prompt_template = optimize_prompt_with_bedrock(BASE_SCHEMA_PROMPT, target_model_id=MODEL_ID)

        # Step 3: Optional evaluation
        if run_quick_eval and eval_examples:
            print("[process_document] Running quick evaluation on provided examples...")
            required_keys = [
                "extracted_invoice_values.patient_name",
                "extracted_invoice_values.total_amount", 
                "extracted_invoice_values.invoice_date",
                "extracted_invoice_values.invoice_number"
            ]
            eval_results = quick_evaluate_prompts(
                BASE_SCHEMA_PROMPT,
                optimized_prompt_template,
                eval_examples,
                required_keys
            )
            print("Quick evaluation results (aggregates):")
            print(json.dumps({
                "original_agg": eval_results["original"]["aggregate"],
                "optimized_agg": eval_results["optimized"]["aggregate"]
            }, indent=2))

        # Step 4: Final LLM call with optimized prompt
        print("[process_document] Sending optimized prompt to model for final extraction...")
        try:
            final_prompt = optimized_prompt_template.format(document_text=document_text)
        except KeyError as e:
            print(f"[process_document] Error formatting optimized prompt: {e}")
            print("[process_document] Falling back to original prompt...")
            final_prompt = BASE_SCHEMA_PROMPT.format(document_text=document_text)
        
        result_text = llm_call_with_converse(final_prompt)
        
        # Try to parse and format as clean JSON
        parsed = safe_extract_json_from_text(result_text)
        if parsed is None:
            print("[process_document] Warning: Could not parse JSON from model output. Returning raw text.")
            return result_text
            
        return json.dumps(parsed, indent=2)
        
    except Exception as e:
        print(f"process_document failed: {e}")
        print(traceback.format_exc())
        raise


# -----------------------
# If run as script - example usage
# -----------------------
if __name__ == '__main__':
    try:
        input_document_path = r"C:\Users\DELL\Desktop\EVEN\invoices\sample-1.png"
        
        # Optional: load evaluation examples
        small_examples = None
        if EVAL_JSONL_PATH:
            small_examples = []
            with open(EVAL_JSONL_PATH, 'r', encoding='utf-8') as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        small_examples.append(json.loads(line))

        json_output = process_document(
            input_document_path, 
            run_quick_eval=bool(small_examples), 
            eval_examples=small_examples
        )
        
        print("\n" + "="*60)
        print("FINAL JSON OUTPUT:")
        print("="*60)
        print(json_output)
        
    except Exception as e:
        print(f"Script failed: {e}")
        print(traceback.format_exc())
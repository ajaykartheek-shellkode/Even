import boto3
import json
import os
from deepdiff import DeepDiff

# ========= Bedrock Client =========
bedrock = boto3.client(
    "bedrock-runtime",
    region_name="ap-south-1"
)

# ========= Configs =========
PROMPT_FOLDER = "prompts"
DIFF_FOLDER = "diff_output"
MAX_ITERS = 2  # stop early if no differences

# ========= Your Schema (unchanged) =========
schema = """
{
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
}
"""

# ========= Your Original Prompt Builder (kept) =========
def build_prompt(schema: str, raw_text: str) -> str:
    return f"""
<INVOICE_EXTRACTION_SYSTEM>
<ROLE>
You are an expert medical and pharmacy invoice data extraction AI. Your task is to analyze invoice text and extract structured information into the specified JSON format.
</ROLE>

<DOCUMENT_TYPES>
You will encounter two main types of invoices:
1. MEDICAL/HOSPITAL INVOICES: Contains medical services, procedures, consultations, diagnostic tests
2. PHARMACY INVOICES: Contains medications, tablets, capsules with MRP, GST, and discount details
</DOCUMENT_TYPES>

<EXTRACTION_GUIDELINES>

<BASIC_INVOICE_INFORMATION>
- doctor_name: Look for "Dr.", "DR.", physician names, doctor signatures
- facility: Extract complete facility name with full address, contact numbers, location details
- invoice_date: Extract date in EXACT original format (DD/MM/YYYY, DD/MM/YYYY HH:MM AM/PM)
- invoice_number: Look for "Invoice No", "Bill No", "Receipt No", numeric identifiers
- patient_name: Extract full patient name, may include titles like "Mr.", "Mrs.", "Ms."
- patient_age: Extract age with units like "29 years", "45 yrs", if not found set to null AND **DO NOT EXTRACT ANYTHING UNNECESSARY TEXT, JUST EXTRACT THE AGE WITH UNITS IF FOUND**
- patient_gender: Extract "Male", "Female", "M", "F", convert to full form
- patient_contact: Extract phone numbers of the patient
- payment_mode: Extract payment methods like "Cash", "Card", "Online"
</BASIC_INVOICE_INFORMATION>

<SERVICES_EXTRACTION>
**CRITICAL: EXTRACT ALL SERVICES/ITEMS - Do not skip any service, medication, or billable item or medicines. Scan the ENTIRE document systematically for every single service entry, including:**
**CRITICAL: ALL medication names, medicines, medical procedures, consultations, diagnostic tests**
**CRITICAL: ALL items in product tables, itemized lists, service sections**
**CRITICAL: ALL pharmacy items with different formulations (tablets, capsules, sprays, injections)**
**CRITICAL: ALL medical services across different departments (radiology, laboratory, consultation, pharmacy)**
**SYSTEMATIC SCANNING: Process line by line through service tables, itemized sections, and billing rows**
For each service/item in the invoice, extract:
- department: Medical department like "radiology", "pharmacy", "consultation", "laboratory"
- service: Complete service name (e.g., "XRAY KNEE JT AP/LAT", "THYRONORM 75MCG TABLET")
- amount: Final calculated amount after all adjustments (as number, not string)
- quantity: Number of units/services provided

</SERVICES_EXTRACTION>

<FINANCIAL_TOTALS>
- total_amount: Final total amount payable
- discount: Total discount amount if mentioned
</FINANCIAL_TOTALS>

<DATA_PROCESSING_RULES>
1. NUMERIC VALUES: Convert all amounts to numbers without currency symbols or commas
2. NULL HANDLING: If field not found - String fields → null, Numeric fields → 0, Arrays → []
3. DEPARTMENT IDENTIFICATION: "radiology", "pharmacy", "consultation", "laboratory", "emergency"
</DATA_PROCESSING_RULES>

<VALIDATION_CHECKS>
Before outputting, verify:
- All numeric fields are actual numbers, not strings
- Date formats are preserved as found in original text
- Service names are complete and accurate
</VALIDATION_CHECKS>

</EXTRACTION_GUIDELINES>

<SCHEMA_DEFINITION>
{schema}
</SCHEMA_DEFINITION>

<INVOICE_TEXT_TO_ANALYZE>
{raw_text}
</INVOICE_TEXT_TO_ANALYZE>

<OUTPUT_INSTRUCTIONS>
CRITICAL REQUIREMENTS:
1. Output ONLY valid JSON matching the schema exactly
2. Do NOT include any explanations, comments, or additional text
3. Do NOT use markdown formatting or code blocks
4. Ensure all numeric values are numbers, not strings
5. Use null for missing string fields, 0 for missing numeric fields
6. Extract ALL available information from the invoice text
7. Preserve original date and text formatting where specified
8. Be thorough and accurate in extraction

RESPOND WITH JSON ONLY:
</OUTPUT_INSTRUCTIONS>

</INVOICE_EXTRACTION_SYSTEM>
"""

# ========= Helpers for Prompt Templates in /prompts =========
def _get_latest_prompt_path(prompt_folder=PROMPT_FOLDER):
    """
    Pick latest improved prompt if exists, else initial_prompt.txt.
    improved_prompt_<n>.txt with largest n wins.
    """
    if not os.path.isdir(prompt_folder):
        return None  # fallback to build_prompt

    improved = []
    for f in os.listdir(prompt_folder):
        if f.startswith("improved_prompt_") and f.endswith(".txt"):
            try:
                n = int(f.split("_")[-1].split(".")[0])
                improved.append((n, f))
            except Exception:
                pass

    if improved:
        improved.sort(key=lambda x: x[0])
        return os.path.join(prompt_folder, improved[-1][1])

    # else initial
    initial_path = os.path.join(prompt_folder, "initial_prompt.txt")
    return initial_path if os.path.exists(initial_path) else None


def _fill_template(template_text: str, schema_text: str, raw_text: str) -> str:
    """
    Safely substitute tokens without .format (which clashes with braces).
    Requires the template to contain {schema} and {raw_text} tokens.
    If not present, return None to trigger fallback to build_prompt.
    """
    if "{schema}" in template_text and "{raw_text}" in template_text:
        return template_text.replace("{schema}", schema_text).replace("{raw_text}", raw_text)
    return None


def _get_next_improved_path(prompt_folder=PROMPT_FOLDER):
    if not os.path.isdir(prompt_folder):
        os.makedirs(prompt_folder, exist_ok=True)

    existing = []
    for f in os.listdir(prompt_folder):
        if f.startswith("improved_prompt_") and f.endswith(".txt"):
            try:
                n = int(f.split("_")[-1].split(".")[0])
                existing.append(n)
            except Exception:
                pass
    next_n = (max(existing) + 1) if existing else 1
    return os.path.join(prompt_folder, f"improved_prompt_{next_n}.txt")

# ========= Your Bedrock Calls (kept) =========
def extract_invoice(raw_text: str):
    """
    Build prompt from latest template in /prompts if available (with {schema} and {raw_text}),
    otherwise fall back to your original build_prompt().
    """
    prompt_path = _get_latest_prompt_path()
    prompt_text = None
    if prompt_path:
        try:
            with open(prompt_path, "r", encoding="utf-8") as pf:
                tpl = pf.read()
            prompt_text = _fill_template(tpl, schema, raw_text)
        except Exception:
            prompt_text = None

    # fallback to your original
    prompt = prompt_text if prompt_text else build_prompt(schema, raw_text)

    try:
        output_text = call_bedrock_llm(prompt)
        return output_text
    except Exception as e:
        return f"Error: {str(e)}"
    

import time
import botocore.exceptions

def call_bedrock_llm(prompt: str, max_retries: int = 5) -> str:
    for attempt in range(max_retries):
        try:
            response = bedrock.converse(
                modelId="apac.anthropic.claude-3-7-sonnet-20250219-v1:0",
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 10000, "temperature": 0.1}
            )
            return response["output"]["message"]["content"][0]["text"]

        except botocore.exceptions.ClientError as e:
            err_code = e.response["Error"]["Code"]
            if err_code == "ThrottlingException":
                print(f"[WARN] Bedrock throttled... waiting 60s before retry (attempt {attempt+1}/{max_retries})")
                time.sleep(60)  # wait exactly 1 minute
            else:
                raise  # bubble up non-throttling errors

    raise RuntimeError("Failed after retries due to repeated ThrottlingException")

# ========= Step 1 (kept) =========
def process_rawtext_files(folder_path="rawtext_data"):
    results = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                raw_text = data.get("raw_text", "")

            print(f"\nExtracting from: {filename}")
            extracted_json = extract_invoice(raw_text)
            results[filename] = extracted_json

    with open("extracted_invoices.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nStep 1 complete: Extracted invoices saved to extracted_invoices.json")
    return results

# ========= Step 2 (kept, with JSON-safe DeepDiff) =========
def compare_extractions(
    extracted_file="extracted_invoices.json",
    expected_folder="expected_output",
    diff_folder=DIFF_FOLDER
):
    if not os.path.exists(extracted_file):
        print("No extracted_invoices.json found. Run extraction first.")
        return
    
    with open(extracted_file, "r", encoding="utf-8") as f:
        extracted_data = json.load(f)

    all_diffs = {}

    for raw_filename, extracted_str in extracted_data.items():
        try:
            extracted_json = json.loads(extracted_str)

            expected_filename = raw_filename.replace("_rawtext.json", "_output.json")
            expected_path = os.path.join(expected_folder, expected_filename)

            if not os.path.exists(expected_path):
                print(f"Expected file missing: {expected_filename}")
                continue

            with open(expected_path, "r", encoding="utf-8") as f:
                expected_json = json.load(f)

            diff = DeepDiff(expected_json, extracted_json, ignore_order=True)

            if diff:
                diff_dict = json.loads(diff.to_json())
                all_diffs[raw_filename] = diff_dict

                print(f"\nDifferences for {raw_filename} vs {expected_filename}:")
                print(json.dumps(diff_dict, indent=2, ensure_ascii=False))
            else:
                print(f"\nMatch: {raw_filename}")

        except Exception as e:
            print(f"Error processing {raw_filename}: {e}")

    with open("differences.json", "w", encoding="utf-8") as f:
        json.dump(all_diffs, f, indent=2, ensure_ascii=False)

    
    # Ensure versioned folder exists
    os.makedirs(diff_folder, exist_ok=True)

    # Find next version number
    existing = []
    for f in os.listdir(diff_folder):
        if f.startswith("deepdiff") and f.endswith(".json"):
            try:
                n = int(f.split("_")[-1].split(".")[0])
                existing.append(n)
            except Exception:
                pass
            
    next_n = (max(existing) + 1) if existing else 1

    # Save per-run snapshot
    diff_path = os.path.join(diff_folder, f"deepdiff_{next_n}.json")
    with open(diff_path, "w", encoding="utf-8") as f:
        json.dump(all_diffs, f, indent=2, ensure_ascii=False)

    print("\nStep 2 complete: Differences saved to differences.json")
    return all_diffs


# ========= Step 3: Improve Prompt (new, minimal addition) =========
def optimize_prompt(differences_file="differences.json", prompt_folder=PROMPT_FOLDER):
    if not os.path.exists(differences_file):
        print("No differences.json found. Run comparison first.")
        return None

    with open(differences_file, "r", encoding="utf-8") as f:
        all_diffs = json.load(f)

    # Load the current (latest) prompt text from /prompts or fallback to initial
    current_path = _get_latest_prompt_path(prompt_folder)
    current_prompt_text = ""
    if current_path and os.path.exists(current_path):
        with open(current_path, "r", encoding="utf-8") as f:
            current_prompt_text = f.read()
    else:
        # Use the internal builder as text baseline if no file exists
        current_prompt_text = build_prompt("{schema}", "{raw_text}")

    # Build optimization instruction
    optimization_context = f"""
You are an expert at prompt engineering for structured data extraction.

The current extraction prompt already has a lot of logic.
Important:
- Do NOT remove any existing instruction unless it directly causes the mistakes shown in the differences.
- You may add clarifications or modify specific parts ONLY where necessary to fix the mismatches.
- Preserve all the original intent and logic.

Current Extraction Prompt Template (may contain tokens like {{schema}} and {{raw_text}}):
{current_prompt_text}

Differences from previous run (DeepDiff JSON):
{json.dumps(all_diffs, indent=2)}

Task:
Return ONLY the improved prompt template (no explanations). Keep tokens like {{schema}} and {{raw_text}} if they already exist.
"""

    try:
        improved_prompt = call_bedrock_llm(optimization_context).strip()
        new_path = _get_next_improved_path(prompt_folder)
        with open(new_path, "w", encoding="utf-8") as f:
            f.write(improved_prompt)
        print(f"\nStep 3 complete: Improved prompt saved to {new_path}")
        return new_path
    except Exception as e:
        print(f"Error optimizing prompt: {e}")
        return None

# ========= Orchestrated Loop (new, minimal addition) =========
def optimize_loop(max_iters=MAX_ITERS):
    """
    Full loop:
      1) Extract with current/latest prompt in /prompts
      2) Compare -> differences.json
      3) If diffs exist -> optimize prompt -> save improved_prompt_<n>.txt
      4) Repeat until no diffs or max_iters
    """
    for i in range(1, max_iters + 1):
        print(f"\n================ Iteration {i} ================")
        process_rawtext_files()
        diffs = compare_extractions()


        if not diffs:
            print("\n✅ No differences found. Stopping.")
            break

        saved_path = optimize_prompt()
        if not saved_path:
            print("\n⚠️ Could not optimize prompt. Stopping.")
            break

    else:
        print("\nSuccessfully Improved the prompt")

# ========= Entry Point =========
if __name__ == "__main__":
    # Run the optimization loop (will re-run extraction after each improved prompt)
    optimize_loop()
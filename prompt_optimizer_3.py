import boto3
import json
import os
from pathlib import Path
from click import prompt
from deepdiff import DeepDiff


bedrock = boto3.client("bedrock-runtime", region_name="ap-south-1")

PROMPT_FOLDER = "prompts"
MAX_ITERS = 5 

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

def extract_invoice(raw_text: str):
    prompt = build_prompt(schema, raw_text)

    params = {
        "modelId": "apac.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "messages": [
            {"role": "user", "content": [{"text": prompt}]}
        ],
        "inferenceConfig": {
            "temperature": 0.0,
            "maxTokens": 10000
        }
    }

    try:
        response = bedrock.converse(**params)
        output_text = response["output"]["message"]["content"][0]["text"]

        return output_text
    except Exception as e:
        return f"Error: {str(e)}"

    
def call_bedrock_llm(prompt: str) -> str:
    response = bedrock.converse(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        messages=[
            {"role": "user", "content": [{"text": prompt}]}
        ],
        inferenceConfig={"maxTokens": 10000, "temperature": 0.1}
    )
    return response["output"]["message"]["content"][0]["text"]


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
    

def compare_extractions(
    extracted_file="extracted_invoices.json",
    expected_folder="expected_output"
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

    print("\nStep 2 complete: Differences saved to differences.json")
    return all_diffs

if __name__ == "__main__":
    process_rawtext_files()

    compare_extractions()
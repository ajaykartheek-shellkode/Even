# ==================================================================================================
# CODE WITH ONLY analyze_expense API - BATCH PROCESSING VERSION
# ==================================================================================================

import boto3
import os
import time
import json
import re
import botocore
from botocore.exceptions import ClientError
import glob
from pathlib import Path

def detect_file_type(file_path):
    """Detect file type based on extension"""
    _, file_extension = os.path.splitext(file_path.lower())
    if file_extension in ['.pdf']:
        return 'pdf'
    elif file_extension in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']:
        return 'image'
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

def get_supported_files_from_folder(folder_path):
    """Get all supported files from a folder"""
    supported_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.tif']
    supported_files = []
    
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    for ext in supported_extensions:
        # Use both uppercase and lowercase patterns
        pattern = str(folder_path / f"*{ext}")
        files = glob.glob(pattern)
        supported_files.extend(files)
        
        pattern_upper = str(folder_path / f"*{ext.upper()}")
        files_upper = glob.glob(pattern_upper)
        supported_files.extend(files_upper)
    
    # Remove duplicates and sort
    supported_files = sorted(list(set(supported_files)))
    
    print(f"Found {len(supported_files)} supported files in folder: {folder_path}")
    for file in supported_files:
        print(f"  - {os.path.basename(file)}")
    
    return supported_files

def upload_file_to_s3(local_file_path, bucket_name, region_name='ap-south-1'):
    """Upload a local file to S3 bucket"""
    try:
        s3_client = boto3.client('s3', region_name=region_name)
        file_name = os.path.basename(local_file_path)
        s3_key = file_name  
        s3_client.upload_file(local_file_path, bucket_name, s3_key)  # Change file_name to s3_key
        print(f"File {local_file_path} uploaded to s3://{bucket_name}/{s3_key}")  # Update print statement
        return s3_key  # Return s3_key instead of file_name
    except ClientError as e:
        print(f"Error uploading file to S3: {e}")
        raise


def extract_text_from_expense_response(expense_documents):
    """Extract text from Textract expense analysis response with Blocks"""
    extracted_text = ""
    
    # with open('expense_documents_385.json', 'w') as f:
    #     json.dump(expense_documents, f, indent=2)
    
    # Initialize categorized data
    customer_info = []
    supplier_info = []
    
    # Process ExpenseDocuments format
    for document in expense_documents:
        
        # Extract RECEIVER/VENDOR specific information from SummaryFields only
        if 'SummaryFields' in document:
            for field in document['SummaryFields']:
                is_receiver = False
                is_vendor = False
                
                # Check GroupProperties for RECEIVER/VENDOR
                if 'GroupProperties' in field:
                    for group_prop in field['GroupProperties']:
                        if 'Types' in group_prop:
                            for type_val in group_prop['Types']:
                                if 'RECEIVER' in type_val.upper():
                                    is_receiver = True
                                if 'VENDOR' in type_val.upper():
                                    is_vendor = True
                
                # Check Type field for RECEIVER/VENDOR
                if 'Type' in field and 'Text' in field['Type']:
                    type_text = field['Type']['Text'].upper()
                    if 'RECEIVER' in type_text:
                        is_receiver = True
                    if 'VENDOR' in type_text:
                        is_vendor = True
                
                # Extract information if it's RECEIVER or VENDOR related
                if is_receiver or is_vendor:
                    label = ""
                    value = ""
                    
                    # Get label from LabelDetection if available
                    if 'LabelDetection' in field and 'Text' in field['LabelDetection']:
                        label = field['LabelDetection']['Text']
                    
                    # Get value from ValueDetection if available
                    if 'ValueDetection' in field and 'Text' in field['ValueDetection']:
                        value = field['ValueDetection']['Text']
                    
                    # If no label, use Type.Text as label
                    if not label and 'Type' in field and 'Text' in field['Type']:
                        label = field['Type']['Text']
                    
                    # Add to appropriate category if we have meaningful data
                    if label and value:
                        info_entry = f"{label}: {value}"
                        if is_receiver:
                            customer_info.append(info_entry)
                        if is_vendor:
                            supplier_info.append(info_entry)
                    elif value and not label:  # Some fields only have value (like pure VENDOR_NAME)
                        if is_receiver:
                            customer_info.append(f"RECEIVER_INFO: {value}")
                        if is_vendor:
                            supplier_info.append(f"VENDOR_INFO: {value}")
    
    # Build the categorized header
    categorized_header = ""
    
    if customer_info:
        categorized_header += "=== CUSTOMER INFORMATION ===\n"
        for info in customer_info:
            categorized_header += f"{info}\n"
        categorized_header += "\n"
    
    if supplier_info:
        categorized_header += "=== SUPPLIER INFORMATION ===\n"
        for info in supplier_info:
            categorized_header += f"{info}\n"
        categorized_header += "\n"
    
    if categorized_header:
        categorized_header += "=== ADDITIONAL DOCUMENT DETAILS ===\n\n"
    
    # Continue with original extraction logic for remaining fields
    for document in expense_documents:
        # Extract summary fields (excluding already processed RECEIVER/VENDOR fields)
        if 'SummaryFields' in document:
            for field in document['SummaryFields']:
                # Skip if this field was already processed as RECEIVER/VENDOR
                skip_field = False
                
                # Check if it's RECEIVER/VENDOR related
                if 'GroupProperties' in field:
                    for group_prop in field['GroupProperties']:
                        if 'Types' in group_prop:
                            for type_val in group_prop['Types']:
                                if 'RECEIVER' in type_val.upper() or 'VENDOR' in type_val.upper():
                                    skip_field = True
                                    break
                        if skip_field:
                            break
                    if skip_field:
                        continue
                
                if 'Type' in field and 'Text' in field['Type']:
                    type_text = field['Type']['Text'].upper()
                    if 'RECEIVER' in type_text or 'VENDOR' in type_text:
                        skip_field = True
                
                if not skip_field:
                    if 'LabelDetection' in field and 'ValueDetection' in field:
                        label = field['LabelDetection'].get('Text', '')
                        value = field['ValueDetection'].get('Text', '')
                        if label and value:
                            extracted_text += f"{label}: {value} "
                    elif 'ValueDetection' in field and 'Type' in field:
                        type_label = field['Type'].get('Text', '')
                        value = field['ValueDetection'].get('Text', '')
                        if type_label and value:
                            extracted_text += f"{type_label}: {value} "
        
        # Extract line item groups (product-level information)
        if 'LineItemGroups' in document:
            for group in document['LineItemGroups']:
                if 'LineItems' in group:
                    for line_item in group['LineItems']:
                        if 'LineItemExpenseFields' in line_item:
                            for field in line_item['LineItemExpenseFields']:
                                if 'LabelDetection' in field and 'ValueDetection' in field:
                                    label = field['LabelDetection'].get('Text', '')
                                    value = field['ValueDetection'].get('Text', '')
                                    if label and value:
                                        extracted_text += f"{label}: {value} "
                            extracted_text += " | "  # Separator between line items
        
        # Extract raw text from blocks
        if 'Blocks' in document:
            extracted_text += "\n\nRaw Text Content:\n" 
            for block in document['Blocks']:
                if block['BlockType'] == 'LINE' and 'Text' in block:
                    extracted_text += block['Text'] + "\n"
    
    # Combine categorized header with extracted text
    final_text = categorized_header + extracted_text.strip()

    # --------------------------------------------------------------------------------------------------------  
    # TEXT WITH CONFIDENCE SCORE  
    # --------------------------------------------------------------------------------------------------------
    # ADD THIS LINE - Extract confidence scores for all text
    # confidence_data = extract_all_text_with_confidence(expense_documents)
    
    # Append confidence data to final text
    # final_text += confidence_data
    # --------------------------------------------------------------------------------------------------------    
    
    return final_text


    
    # with open('final_extracted_output_385.txt', 'w', encoding='utf-8') as file:
    #   file.write(final_text)
    
    # return final_text

def process_image_with_textract(bucket_name, file_name, region_name='ap-south-1'):
    """Process image files (PNG, JPEG, etc.) with Textract expense analysis"""
    try:
        # Start timing for image expense analysis
        start_time = time.time()
        print(f"Starting image expense analysis at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        
        textract_client = boto3.client('textract', region_name=region_name)
        
        response = textract_client.analyze_expense(
            Document={
                'S3Object': {
                    'Bucket': bucket_name,
                    'Name': file_name
                }
            }
        )

        # print(f"🔴🔴🔴🔴🔴🔴 ----- response getting saved ")
        # with open("analyze_expense_json.json", "w", encoding="utf-8") as file:
        #   json.dump(response, file, indent=2, ensure_ascii=False)

        # End timing for image expense analysis
        end_time = time.time()
        duration = end_time - start_time
        print(f"Image expense analysis completed at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"analyze_expense API (image) took: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
        print("Expense analysis completed for image")
        
        # Extract text from the structured expense response
        if 'ExpenseDocuments' in response:
            extracted_text = extract_text_from_expense_response(response['ExpenseDocuments'])
        else:
            print("No expense documents found in response")
            extracted_text = ""
        
        return extracted_text, response
    except ClientError as e:
        print(f"Textract error processing image with expense analysis: {e}")
        raise

def process_pdf_with_textract(bucket_name, file_name, region_name='ap-south-1'):
    """Process PDF files with Textract expense analysis asynchronously"""
    try:
        # Start timing for PDF expense analysis
        start_time = time.time()
        print(f"Starting PDF expense analysis at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        
        textract_client = boto3.client('textract', region_name=region_name)
        
        # Start asynchronous expense analysis job
        job_start_time = time.time()
        response = textract_client.start_expense_analysis(
            DocumentLocation={
                'S3Object': {
                    'Bucket': bucket_name,
                    'Name': file_name
                }
            }
        )
        
        print(f"Response from start_expense_analysis: {response}")

        job_id = response['JobId']
        print(f"Started Textract expense analysis job {job_id} for PDF processing")
        
        # Wait for the job to complete
        while True:
            response = textract_client.get_expense_analysis(JobId=job_id)

            status = response['JobStatus']
            print(f"Job status: {status}")
            
            if status in ['SUCCEEDED', 'FAILED']:
                break
                
            time.sleep(5)  # Wait for 5 seconds before checking again
        
        if status == 'FAILED':
            print(f"Textract expense analysis job failed: {response.get('StatusMessage', 'Unknown error')}")
            raise Exception("PDF expense analysis failed")
        
        # Job completion timing
        job_end_time = time.time()
        job_duration = job_end_time - job_start_time
        print(f"Textract job processing took: {job_duration:.2f} seconds ({job_duration/60:.2f} minutes)")
        
        # Collect expense data from all pages
        all_expense_documents = []
        pages = [response]
        
        # Handle pagination for large PDFs
        next_token = response.get('NextToken')
        print(f"Next token: {next_token}")

        while next_token:
            response = textract_client.get_expense_analysis(
                JobId=job_id,
                NextToken=next_token
            )
            pages.append(response)
            next_token = response.get('NextToken')
        
        # Collect all expense documents from all pages
        for page in pages:
            if 'ExpenseDocuments' in page:
                all_expense_documents.extend(page['ExpenseDocuments'])
        
        # Extract text from all expense documents
        if all_expense_documents:
            extracted_text = extract_text_from_expense_response(all_expense_documents)
        else:
            print("No expense documents found in any page")
            extracted_text = ""
        
        # End timing for complete PDF expense analysis
        end_time = time.time()
        total_duration = end_time - start_time
        print(f"PDF expense analysis completed at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"Total PDF expense analysis took: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        
        return extracted_text, {'ExpenseDocuments': all_expense_documents}
    except ClientError as e:
        print(f"Textract error processing PDF with expense analysis: {e}")
        raise

def extract_text_from_document(file_path=None, bucket_name=None, file_name=None, region_name='ap-south-1'):
    """
    Extract text from a document (PDF or image) using Textract expense analysis
    Input can be either a local file path or an S3 location
    """
    
    # If local file is provided, upload it to S3 first
    if file_path:
        if not bucket_name:
            raise ValueError("S3 bucket name must be provided for file upload")
        file_name = upload_file_to_s3(file_path, bucket_name, region_name)
    
    # Make sure we have both bucket and file name at this point
    if not bucket_name or not file_name:
        raise ValueError("Both bucket_name and file_name must be provided")
    
    # Detect file type and process accordingly
    file_type = detect_file_type(file_name)
    print(f"Detected file type: {file_type}")
    
    if file_type == 'image':
        extracted_text, textract_response = process_image_with_textract(bucket_name, file_name, region_name)
    elif file_type == 'pdf':
        extracted_text, textract_response = process_pdf_with_textract(bucket_name, file_name, region_name)
    
    return extracted_text, textract_response

# =====================================================================================================================
# FUNCTION FOR EXTRACTED TEXT ALONG WITH THE CONFIDENCE SCORE
# =====================================================================================================================

def extract_all_text_with_confidence(expense_documents):
    """Extract all text with confidence scores from Textract expense response"""
    confidence_text = "\n\n=== EXTRACTED TEXT WITH CONFIDENCE SCORES ===\n\n"
    
    for doc_idx, document in enumerate(expense_documents):
        confidence_text += f"--- Document {doc_idx + 1} ---\n\n"
        
        # Extract from SummaryFields
        if 'SummaryFields' in document:
            confidence_text += "SUMMARY FIELDS:\n"
            for field_idx, field in enumerate(document['SummaryFields']):
                confidence_text += f"  Field {field_idx + 1}:\n"
                
                # Type field
                if 'Type' in field and 'Text' in field['Type']:
                    type_text = field['Type']['Text']
                    type_conf = field['Type'].get('Confidence', 0)
                    confidence_text += f"    Type: {type_text} (Confidence: {type_conf:.2f})\n"
                
                # LabelDetection field
                if 'LabelDetection' in field and 'Text' in field['LabelDetection']:
                    label_text = field['LabelDetection']['Text']
                    label_conf = field['LabelDetection'].get('Confidence', 0)
                    confidence_text += f"    Label: {label_text} (Confidence: {label_conf:.2f})\n"
                
                # ValueDetection field
                if 'ValueDetection' in field and 'Text' in field['ValueDetection']:
                    value_text = field['ValueDetection']['Text']
                    value_conf = field['ValueDetection'].get('Confidence', 0)
                    confidence_text += f"    Value: {value_text} (Confidence: {value_conf:.2f})\n"
                
                confidence_text += "\n"
        
        # Extract from LineItemGroups
        if 'LineItemGroups' in document:
            confidence_text += "LINE ITEM GROUPS:\n"
            for group_idx, group in enumerate(document['LineItemGroups']):
                confidence_text += f"  Group {group_idx + 1}:\n"
                
                if 'LineItems' in group:
                    for item_idx, line_item in enumerate(group['LineItems']):
                        confidence_text += f"    Line Item {item_idx + 1}:\n"
                        
                        if 'LineItemExpenseFields' in line_item:
                            for field_idx, field in enumerate(line_item['LineItemExpenseFields']):
                                confidence_text += f"      Field {field_idx + 1}:\n"
                                
                                # Type field
                                if 'Type' in field and 'Text' in field['Type']:
                                    type_text = field['Type']['Text']
                                    type_conf = field['Type'].get('Confidence', 0)
                                    confidence_text += f"        Type: {type_text} (Confidence: {type_conf:.2f})\n"
                                
                                # LabelDetection field
                                if 'LabelDetection' in field and 'Text' in field['LabelDetection']:
                                    label_text = field['LabelDetection']['Text']
                                    label_conf = field['LabelDetection'].get('Confidence', 0)
                                    confidence_text += f"        Label: {label_text} (Confidence: {label_conf:.2f})\n"
                                
                                # ValueDetection field
                                if 'ValueDetection' in field and 'Text' in field['ValueDetection']:
                                    value_text = field['ValueDetection']['Text']
                                    value_conf = field['ValueDetection'].get('Confidence', 0)
                                    confidence_text += f"        Value: {value_text} (Confidence: {value_conf:.2f})\n"
                        
                        confidence_text += "\n"
        
        # Extract from Blocks (LINE and WORD level)
        if 'Blocks' in document:
            confidence_text += "BLOCKS (Raw OCR Text):\n"
            
            # LINE level blocks
            line_blocks = [block for block in document['Blocks'] if block['BlockType'] == 'LINE']
            if line_blocks:
                confidence_text += "  LINE Blocks:\n"
                for line_idx, block in enumerate(line_blocks):
                    if 'Text' in block:
                        line_text = block['Text']
                        line_conf = block.get('Confidence', 0)
                        confidence_text += f"    Line {line_idx + 1}: {line_text} (Confidence: {line_conf:.2f})\n"
            
            # WORD level blocks
            word_blocks = [block for block in document['Blocks'] if block['BlockType'] == 'WORD']
            if word_blocks:
                confidence_text += "\n  WORD Blocks:\n"
                for word_idx, block in enumerate(word_blocks):
                    if 'Text' in block:
                        word_text = block['Text']
                        word_conf = block.get('Confidence', 0)
                        confidence_text += f"    Word {word_idx + 1}: {word_text} (Confidence: {word_conf:.2f})\n"
        
        confidence_text += "\n" + "="*50 + "\n\n"
    
    return confidence_text


def parse_invoice_with_llm(raw_text, confidence_scores=None, region_name='ap-south-1'):
    """Parse invoice text using Bedrock LLM to extract structured data (using Converse API)"""
    try:

        # Configure boto3 clients
        config = botocore.config.Config(
            read_timeout=1800,
            connect_timeout=1800,
            retries={'max_attempts': 10}
        )

        # Initialize the Bedrock runtime client
        bedrock_runtime = boto3.client('bedrock-runtime', region_name=region_name, config=config)

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

        prompt = f"""
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


        # Set up parameters for converse API
        params = {
            "modelId": "apac.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {
                "temperature": 0.0,
                "maxTokens": 10000
            }
        }
        
        # Make the API call using converse
        response = bedrock_runtime.converse(**params)
        
        # Extract the response text
        model_output = response['output']['message']['content'][0]['text']
        
        # Clean up the output to ensure it's valid JSON
        json_output = model_output.strip()
        
        # Use regex to find JSON object (anything between { and })
        json_match = re.search(r'(\{.*\})', json_output, re.DOTALL)
        if json_match:
            json_output = json_match.group(1)
        
        # Parse the JSON to verify it's valid
        try:
            parsed_data = json.loads(json_output)
            # Merge actual confidence scores if provided
            if confidence_scores and 'confidenceScores' in parsed_data:
                parsed_data['confidenceScores'] = confidence_scores
            return parsed_data
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON output: {e}")
            print(f"Raw LLM output: {model_output}")
            
            # Try again with more aggressive JSON extraction
            try:
                # Find the position of the first '{' and last '}'
                start_pos = model_output.find('{')
                end_pos = model_output.rfind('}')
                
                if start_pos != -1 and end_pos != -1 and end_pos > start_pos:
                    json_output = model_output[start_pos:end_pos+1]
                    return json.loads(json_output)
                else:
                    return {"error": "Could not parse the extracted data", "raw_text": raw_text}
            except:
                return {"error": "Could not parse the extracted data", "raw_text": raw_text}
    
    except ClientError as e:
        print(f"Bedrock API error: {e}")
        raise
    except Exception as e:
        print(f"Error in LLM processing: {str(e)}")
        raise


def process_single_file(file_path, bucket_name, region_name, output_folder):
    """Process a single file and generate JSON output"""
    try:
        print(f"\n{'='*80}")
        print(f"Processing file: {os.path.basename(file_path)}")
        print(f"{'='*80}")
        
        # Extract text from document
        extracted_text, textract_response = extract_text_from_document(
            file_path=file_path, 
            bucket_name=bucket_name, 
            region_name=region_name
        )
        
        print(f"Text extraction completed for: {os.path.basename(file_path)}")
        
        # Parse the invoice data
        print("Parsing invoice data with LLM...")
        invoice_data = parse_invoice_with_llm(extracted_text, region_name=region_name)
        
        # Add raw_text to the output JSON
        invoice_data['raw_text'] = extracted_text
        
        # Generate output filename
        file_basename = Path(file_path).stem  # Get filename without extension
        output_filename = f"{file_basename}_output.json"
        output_path = os.path.join(output_folder, output_filename)
        
        # Save JSON output
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(invoice_data, f, indent=2, ensure_ascii=False)
        
        print(f"JSON output saved: {output_path}")
        
        return {
            'file_path': file_path,
            'status': 'success',
            'output_path': output_path,
            'invoice_data': invoice_data
        }
        
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return {
            'file_path': file_path,
            'status': 'error',
            'error': str(e)
        }


def main():
    # Configuration
    bucket_name = "even-ocr-poc"
    region_name = "ap-south-1"
    
    # Folder containing the files to process
    input_folder = r"C:\Users\sajay\OneDrive\Desktop\Shellkode\Even-V4\input-data"  # Change this to your folder path
    
    # Output folder for JSON files
    output_folder = r"C:\Users\sajay\OneDrive\Desktop\Shellkode\Even-V4\extracted-output"  # Change this to your desired output folder
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Starting batch processing of invoices")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"S3 bucket: {bucket_name}")
    print(f"Region: {region_name}")
    
    try:
        # Get all supported files from the folder
        supported_files = get_supported_files_from_folder(input_folder)
        
        if not supported_files:
            print("No supported files found in the folder!")
            return []
        
        results = []
        successful_count = 0
        failed_count = 0
        
        # Process each file
        for i, file_path in enumerate(supported_files, 1):
            print(f"\nProcessing file {i}/{len(supported_files)}")
            
            result = process_single_file(file_path, bucket_name, region_name, output_folder)
            results.append(result)
            
            if result['status'] == 'success':
                successful_count += 1
            else:
                failed_count += 1
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"BATCH PROCESSING SUMMARY")
        print(f"{'='*80}")
        print(f"Successfully processed: {successful_count} files")
        print(f"Failed to process: {failed_count} files")
        print(f"Output folder: {output_folder}")
        
        if failed_count > 0:
            print(f"\nFailed files:")
            for result in results:
                if result['status'] == 'error':
                    print(f"  - {os.path.basename(result['file_path'])}: {result['error']}")
        
        return results
        
    except Exception as e:
        print(f"Error in batch processing: {str(e)}")
        return []


if __name__ == "__main__":
    results = main()
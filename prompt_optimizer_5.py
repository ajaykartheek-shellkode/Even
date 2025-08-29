import os
import json
import glob
from pathlib import Path
from deepdiff import DeepDiff
import boto3


# =========================
# 1. JSON validation logic
# =========================
def validate_json(expected_json_str, predicted_json_str):
    try:
        expected = json.loads(expected_json_str)
        predicted = json.loads(predicted_json_str)
    except Exception as e:
        return {"valid": False, "error": f"Invalid JSON: {e}"}

    diff = DeepDiff(expected, predicted, ignore_order=True)

    feedback = []
    if diff:
        if "dictionary_item_added" in diff:
            feedback.append("Extra fields found in prediction.")
        if "dictionary_item_removed" in diff:
            feedback.append("Missing fields in prediction.")
        if "type_changes" in diff:
            feedback.append("Some fields have incorrect types.")
        if "values_changed" in diff:
            feedback.append("Some values are incorrect.")

    return {"valid": len(diff) == 0, "diff": diff, "feedback": feedback}


# =========================
# 2. Prompt builder
# =========================
def build_prompt(schema: str, raw_text: str) -> str:
    template = Path("initial_prompt.txt").read_text(encoding="utf-8")
    return template.replace("{schema}", schema).replace("{raw_text}", raw_text)


# =========================
# 3. Run Bedrock LLM
# =========================
def run_bedrock(prompt: str, model_id="anthropic.claude-3-sonnet-20240229-v1:0") -> str:
    client = boto3.client("bedrock-runtime", region_name="ap-south-1")
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps({
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            "max_tokens": 2000,
        }),
    )
    resp_body = json.loads(response["body"].read())
    return resp_body["output"]["message"]["content"][0]["text"]


# =========================
# 4. Extract invoice
# =========================
def extract_invoice(raw_text: str, schema: str, prompt_file: str) -> dict:
    prompt_template = Path(prompt_file).read_text(encoding="utf-8")
    filled_prompt = prompt_template.replace("{schema}", schema).replace("{raw_text}", raw_text)

    output = run_bedrock(filled_prompt)

    try:
        return json.loads(output)
    except Exception:
        return {"error": "Invalid JSON output", "raw_output": output}


# =========================
# 5. Diff filtering
# =========================
def _filter_diffs(all_diffs):
    filtered = {}
    for fname, diff in all_diffs.items():
        clean = {}
        for key, issues in diff.items():
            if key == "type_changes":
                for path, detail in issues.items():
                    # skip harmless int ↔ NoneType diffs
                    if not (detail["old_type"] == "int" and detail["new_type"] == "NoneType"):
                        clean.setdefault("type_changes", {})[path] = detail
            else:
                clean[key] = issues
        if clean:
            filtered[fname] = clean
    return filtered


# =========================
# 6. Prompt optimizer
# =========================
def optimize_prompt(all_diffs, current_prompt_text):
    # filter noise from diffs
    all_diffs = _filter_diffs(all_diffs)

    prompt = f"""
You are an expert prompt engineer.
Your job is to **improve the given prompt** so that the model produces correct JSON.

Current Prompt Template:
------------------------
{current_prompt_text}

Observed Mistakes (DeepDiff summary):
-------------------------------------
{json.dumps(all_diffs, indent=2)}

Optimization Instructions:
--------------------------
- Modify ONLY the parts of the prompt needed to fix the mistakes.
- If differences are about null vs 0 or int vs NoneType, IGNORE them.
- Do NOT add new instructions unless absolutely necessary.
- Do NOT remove existing instructions unless they directly cause errors.
- Keep placeholders {{schema}} and {{raw_text}} exactly as they are (do not replace with actual data).
- Output ONLY the improved prompt text (no explanations, no markdown).
"""

    improved_prompt = run_bedrock(prompt)
    return improved_prompt.strip()


# =========================
# 7. Iterative pipeline
# =========================
def iterative_pipeline(schema_file, raw_folder, expected_folder, iterations=3):
    schema = Path(schema_file).read_text(encoding="utf-8")

    # Start from template
    base_prompt_file = "prompts/initial_prompt.txt"
    current_prompt_text = Path(base_prompt_file).read_text(encoding="utf-8")

    for i in range(1, iterations + 1):
        print(f"\n--- Iteration {i} ---")

        all_diffs = {}
        raw_files = glob.glob(os.path.join(raw_folder, "*.txt"))

        for raw_file in raw_files:
            fname = Path(raw_file).stem
            raw_text = Path(raw_file).read_text(encoding="utf-8")
            expected_file = os.path.join(expected_folder, f"{fname}.json")

            if not os.path.exists(expected_file):
                continue

            expected_json_str = Path(expected_file).read_text(encoding="utf-8")
            predicted_json = extract_invoice(raw_text, schema, base_prompt_file)
            predicted_json_str = json.dumps(predicted_json)

            result = validate_json(expected_json_str, predicted_json_str)
            if not result["valid"]:
                all_diffs[fname] = result["diff"]

        if not all_diffs:
            print("✅ No differences found. Stopping optimization.")
            break

        improved_prompt = optimize_prompt(all_diffs, current_prompt_text)

        # Save improved template
        improved_prompt_file = f"improved_prompt_{i}.txt"
        Path(improved_prompt_file).write_text(improved_prompt, encoding="utf-8")
        print(f"Saved: {improved_prompt_file}")

        # Use new prompt for next round
        base_prompt_file = improved_prompt_file
        current_prompt_text = improved_prompt


# =========================
# 8. Main
# =========================
if __name__ == "__main__":
    iterative_pipeline(
        schema_file="schema.json",
        raw_folder="raw",
        expected_folder="expected",
        iterations=5,
    )

#!/usr/bin/env python3
"""
PromptOptimizer v4 — Patch-mode improvement (fixed NameError)

Fix:
- Build the model instruction string using concatenation (not f-strings) so literal braces
  like {schema} and {raw_text} are preserved and not interpreted by Python.

Other behavior unchanged (patch-mode, canary testing, schema enforcement, etc).
"""

import json
import boto3
import time
import os
import logging
import traceback
from random import uniform
from deepdiff import DeepDiff
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------- Helper utilities ----------------
def safe_num(s):
    if s is None:
        return 0
    if isinstance(s, (int, float)):
        return s
    try:
        t = str(s).strip().replace(",", "").replace("₹", "").replace("$", "")
        if t == "":
            return 0
        if "." in t:
            return float(t)
        return int(t)
    except Exception:
        try:
            return float(str(s).strip())
        except Exception:
            return 0

def ensure_schema_compliance(parsed: Optional[dict], schema_template: dict) -> dict:
    if parsed is None:
        parsed = {}

    out = {}
    ev = parsed.get("extracted_invoice_values", {})
    keys_string = ["invoice_number","patient_name","doctor_name","facility","invoice_date","payment_mode","patient_age","patient_gender","patient_contact","discount"]
    keys_numeric = ["total_amount"]
    for k in keys_string:
        out[k] = ev.get(k) if ev.get(k) not in ("", None) else None
    for k in keys_numeric:
        out[k] = safe_num(ev.get(k))
    services = ev.get("services") if isinstance(ev.get("services"), list) else []
    normalized_services = []
    for s in services:
        if s is None:
            continue
        service_name = s.get("service") if isinstance(s.get("service"), str) else None
        amount = safe_num(s.get("amount")) if s.get("amount") is not None else 0
        quantity = safe_num(s.get("quantity")) if s.get("quantity") is not None else 0
        department = s.get("department") if isinstance(s.get("department"), str) else None
        normalized_services.append({"service": service_name, "amount": amount, "quantity": quantity, "department": department})
    out["services"] = normalized_services
    return {"extracted_invoice_values": out}

def per_field_report(expected: dict, actual: dict) -> Dict[str, bool]:
    report = {}
    e = expected.get("extracted_invoice_values", {})
    a = actual.get("extracted_invoice_values", {})
    report["invoice_number"] = (e.get("invoice_number") == a.get("invoice_number"))
    report["total_amount"] = abs(safe_num(e.get("total_amount")) - safe_num(a.get("total_amount"))) < 1e-3 if (e.get("total_amount") is not None or a.get("total_amount") is not None) else True
    report["services_count"] = len(e.get("services", [])) == len(a.get("services", []))
    exp_names = [ (s.get("service") or "").lower() for s in e.get("services", []) ]
    act_names = [ (s.get("service") or "").lower() for s in a.get("services", []) ]
    report["services_names_subset"] = all(any(en in an for an in act_names) for en in exp_names) if exp_names else True
    report["patient_name"] = ( (e.get("patient_name") or "").strip() == (a.get("patient_name") or "").strip() ) if e.get("patient_name") else True
    return report

# ---------------- PromptOptimizer ----------------
class PromptOptimizer:
    PROMPT_START = "<<<PROMPT_START>>>"
    PROMPT_END = "<<<PROMPT_END>>>"
    JSON_START = "<<<JSON_START>>>"
    JSON_END = "<<<JSON_END>>>"

    def __init__(self, region_name="ap-south-1", model_id="apac.anthropic.claude-3-7-sonnet-20250219-v1:0", max_tokens=4096, results_dir="results"):
        self.bedrock_client = boto3.client("bedrock-runtime", region_name=region_name)
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.api_calls = 0
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "prompts").mkdir(parents=True, exist_ok=True)
        (self.results_dir / "extractions").mkdir(parents=True, exist_ok=True)

        few_shot = """
EXAMPLES:
Example 1:
RAW_TEXT: "Invoice No: 12345\nPatient: John Doe\nService: XRAY KNEE - 300\nTotal: 300"
OUTPUT:
<<<JSON_START>>>
{"extracted_invoice_values": {"invoice_number":"12345","patient_name":"John Doe","services":[{"service":"XRAY KNEE","amount":300,"quantity":1,"department":"radiology"}],"total_amount":300,"doctor_name":null,"facility":null,"invoice_date":null,"payment_mode":null,"patient_age":null,"patient_gender":null,"patient_contact":null,"discount":0}}
<<<JSON_END>>>

Example 2:
RAW_TEXT: "Bill No: INV-987\nPatient: Ms. Priya\nMedicine: AMOXICILLIN 500MG 10 120\nTotal Payable: 120"
OUTPUT:
<<<JSON_START>>>
{"extracted_invoice_values": {"invoice_number":"INV-987","patient_name":"Ms. Priya","services":[{"service":"AMOXICILLIN 500MG","amount":120,"quantity":10,"department":"pharmacy"}],"total_amount":120,"doctor_name":null,"facility":null,"invoice_date":null,"payment_mode":null,"patient_age":null,"patient_gender":null,"patient_contact":null,"discount":0}}
<<<JSON_END>>>
"""

        # use double braces to keep {schema} and {raw_text} exact for later replace
        self.base_prompt = (
            "<INVOICE_EXTRACTION_SYSTEM>\n"
            "ROLE: You are an expert medical and pharmacy invoice data extraction AI. Extract into JSON exactly matching schema.\n\n"
            f"{few_shot}\n"
            "SCHEMA:\n"
            "{{schema}}\n\n"
            "INPUT:\n"
            "{{raw_text}}\n\n"
            "OUTPUT RULES:\n"
            f"- Output JSON only, wrapped between {self.JSON_START} and {self.JSON_END}.\n"
            "- Use null for missing strings, 0 for missing numbers, arrays empty list.\n"
            "- All amounts must be numeric (no commas/currency).\n"
            "- Do not add commentary.\n"
            "</INVOICE_EXTRACTION_SYSTEM>\n"
        )

        self.schema_template = {
            "extracted_invoice_values": {
                "invoice_number": "",
                "patient_name": "",
                "services": [{"service": "", "amount": 0, "quantity": 0, "department": ""}],
                "total_amount": 0,
                "doctor_name": "",
                "facility": "",
                "invoice_date": "",
                "payment_mode": "",
                "patient_age": "",
                "patient_gender": "",
                "patient_contact": "",
                "discount": 0
            }
        }

        self.schema_str = json.dumps(self.schema_template, indent=2)
        self.current_prompt = self.base_prompt

    # ---------------- Bedrock call ----------------
    def call_bedrock(self, prompt: str, max_retries=4) -> Optional[str]:
        for attempt in range(max_retries):
            try:
                params = {
                    "modelId": self.model_id,
                    "messages": [{"role": "user", "content": [{"text": prompt}]}],
                    "inferenceConfig": {"temperature": 0.0, "maxTokens": self.max_tokens}
                }
                resp = self.bedrock_client.converse(**params)
                self.api_calls += 1
                try:
                    text = resp["output"]["message"]["content"][0]["text"]
                except Exception:
                    text = self._find_first_string(resp)
                time.sleep(1)
                return text
            except Exception as e:
                s = str(e).lower()
                logger.warning(f"Bedrock call error: {e}")
                if any(x in s for x in ("throttl","rate","429","too many tokens","504")):
                    if attempt < max_retries - 1:
                        wait = (2**attempt) + uniform(0,1)
                        logger.info(f"Backoff {wait:.1f}s (attempt {attempt+1}/{max_retries})")
                        time.sleep(wait)
                        continue
                    else:
                        logger.error("Max retries for bedrock reached.")
                        return None
                else:
                    return None
        return None

    def _find_first_string(self, obj):
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            for v in obj.values():
                r = self._find_first_string(v)
                if r:
                    return r
        if isinstance(obj, list):
            for v in obj:
                r = self._find_first_string(v)
                if r:
                    return r
        return None

    # ---------------- JSON extraction ----------------
    def find_balanced_json(self, text: str) -> Optional[str]:
        if not text or not isinstance(text, str):
            return None
        if self.JSON_START in text and self.JSON_END in text:
            s = text.index(self.JSON_START) + len(self.JSON_START)
            e = text.index(self.JSON_END, s)
            return text[s:e].strip()
        for opening, closing in [("{","}"),("[","]")]:
            i = text.find(opening)
            if i == -1:
                continue
            depth = 0
            for j in range(i, len(text)):
                ch = text[j]
                if ch == opening:
                    depth += 1
                elif ch == closing:
                    depth -= 1
                    if depth == 0:
                        return text[i:j+1]
        return None

    def extract_json_from_text(self, text: str) -> Optional[dict]:
        cand = self.find_balanced_json(text)
        if not cand:
            return None
        try:
            return json.loads(cand)
        except Exception:
            first = cand.find("{")
            last = cand.rfind("}")
            if first != -1 and last != -1 and last > first:
                try:
                    return json.loads(cand[first:last+1])
                except Exception:
                    return None
            return None

    # ---------------- Scoring ----------------
    def calculate_accuracy(self, expected: dict, actual_raw: str) -> float:
        actual_parsed = self.extract_json_from_text(actual_raw)
        if actual_parsed is None:
            return 0.0
        actual_norm = ensure_schema_compliance(actual_parsed, self.schema_template)
        expected_norm = ensure_schema_compliance({"extracted_invoice_values": expected}, self.schema_template)
        diff = DeepDiff(expected_norm, actual_norm, ignore_order=True).to_dict()
        if not diff:
            return 1.0
        penalty = 0.0
        for k, v in diff.items():
            count = len(v) if isinstance(v, (list, dict)) else 1
            base = 0.12
            try:
                v_text = str(v).lower()
            except Exception:
                v_text = ""
            if "services" in v_text or "total_amount" in v_text or "amount" in v_text:
                base = 0.3
            if k in ("type_changes","values_changed"):
                base *= 1.5
            penalty += base * count
        accuracy = max(0.0, 1.0 - penalty)
        return accuracy

    # ---------------- Patch apply ----------------
    def apply_patch_ops(self, original: str, ops: List[dict]) -> Optional[str]:
        cur = original
        try:
            for op in ops:
                if not isinstance(op, dict) or "op" not in op:
                    logger.error("Invalid op format, skipping patch.")
                    return None
                name = op.get("op")
                if name == "replace":
                    old = op.get("old", "")
                    new = op.get("new", "")
                    if old == "":
                        logger.error("replace op missing 'old'")
                        return None
                    cur = cur.replace(old, new, 1)
                elif name == "replace_all":
                    old = op.get("old", "")
                    new = op.get("new", "")
                    if old == "":
                        return None
                    cur = cur.replace(old, new)
                elif name == "insert_before":
                    match = op.get("match", "")
                    text = op.get("text", "")
                    idx = cur.find(match)
                    if idx == -1:
                        logger.error(f"insert_before match not found: {match}")
                        return None
                    cur = cur[:idx] + text + cur[idx:]
                elif name == "insert_after":
                    match = op.get("match", "")
                    text = op.get("text", "")
                    idx = cur.find(match)
                    if idx == -1:
                        logger.error(f"insert_after match not found: {match}")
                        return None
                    idx_end = idx + len(match)
                    cur = cur[:idx_end] + text + cur[idx_end:]
                elif name == "prepend":
                    cur = op.get("text", "") + cur
                elif name == "append":
                    cur = cur + op.get("text", "")
                elif name == "delete":
                    match = op.get("match", "")
                    if match == "":
                        return None
                    cur = cur.replace(match, "", 1)
                else:
                    logger.error(f"Unknown op: {name}")
                    return None
            return cur
        except Exception:
            logger.exception("Failed applying patch ops")
            return None

    # ---------------- Patch improvement (fixed instruction build) ----------------
    def improve_prompt_with_patch(self, low_cases: List[dict], max_canaries=3, min_delta=0.04) -> Optional[str]:
        op_format_example = {
            "ops": [
                {"op": "replace", "old": "OLD TEXT", "new": "NEW TEXT"},
                {"op": "insert_before", "match": "SOME_MARKER", "text": "INSERTED LINE\n"},
                {"op": "append", "text": "\n# extra note (must not be commentary)"},
            ]
        }

        cases_text = ""
        for c in low_cases[:4]:
            cases_text += "\nFILE: " + c['file'] + "\nACCURACY: " + f"{c['accuracy']:.3f}" + "\nRAW_TEXT (truncated):\n" + c['raw_text'][:1500] + "\nEXPECTED:\n" + json.dumps(c['expected'], indent=2)[:1500] + "\n---\n"

        # Build instruction with concatenation so braces are preserved literally
        instruction = (
            "You are a prompt engineer. The current extraction prompt sometimes fails. Produce a SMALL, PRECISE PATCH describing edits to the current prompt to fix consistent extraction errors.\n\n"
            "Return ONLY a JSON object (no commentary) wrapped between these delimiters exactly:\n\n"
            + self.PROMPT_START + "\n"
            "{\n"
            '  "ops": [\n'
            "    // list of op objects (see examples)\n"
            "  ]\n"
            "}\n"
            + self.PROMPT_END + "\n\n"
            "Supported op types and behaviors:\n"
            "- replace: replace first occurrence of \"old\" with \"new\"\n"
            "- replace_all: replace all occurrences of \"old\" with \"new\"\n"
            "- insert_before: find \"match\" and insert \"text\" before it\n"
            "- insert_after: find \"match\" and insert \"text\" after it\n"
            "- prepend: add \"text\" at the beginning\n"
            "- append: add \"text\" at the end\n"
            "- delete: delete first occurrence of \"match\"\n\n"
            "IMPORTANT:\n"
            "- The ops must not remove the placeholders \"{schema}\" or \"{raw_text}\".\n"
            "- The ops must not remove the JSON delimiters " + self.JSON_START + "/" + self.JSON_END + ".\n"
            "- Keep the patch minimal. Do NOT return a full new prompt.\n"
            "- Ensure the patch text does not include unescaped braces that would break Python .replace uses.\n\n"
            "Example format (for reference only):\n"
            + json.dumps(op_format_example, indent=2) + "\n\n"
            "Context — current prompt (for reference):\n\n"
            + (self.current_prompt[:3000]) + "\n\n"
            "Failed examples:\n\n"
            + cases_text
        )

        logger.info("Requesting patch ops from model...")
        resp = self.call_bedrock(instruction)
        if not resp:
            logger.error("No response for patch request")
            return None

        candidate_json = None
        try:
            start = resp.index(self.PROMPT_START) + len(self.PROMPT_START)
            end = resp.index(self.PROMPT_END, start)
            candidate_json = resp[start:end].strip()
        except ValueError:
            candidate_json = resp.strip()

        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        (self.results_dir / "prompts" / f"patch_candidate_{ts}.txt").write_text(candidate_json, encoding="utf-8")

        try:
            parsed = json.loads(candidate_json)
        except Exception:
            logger.error("Candidate patch is not valid JSON; rejecting")
            return None

        ops = parsed.get("ops") if isinstance(parsed, dict) else None
        if not ops or not isinstance(ops, list):
            logger.error("Candidate JSON missing 'ops' list; rejecting")
            return None

        candidate_prompt = self.apply_patch_ops(self.current_prompt, ops)
        if candidate_prompt is None:
            logger.error("Failed to apply patch ops; rejecting")
            return None

        if "{schema}" not in candidate_prompt or "{raw_text}" not in candidate_prompt:
            logger.error("Candidate prompt lost required placeholders; rejecting")
            return None
        if self.JSON_START not in candidate_prompt or self.JSON_END not in candidate_prompt:
            logger.error("Candidate prompt lost JSON delimiters; rejecting")
            return None
        if len(candidate_prompt.strip()) < 150:
            logger.error("Candidate prompt too short after applying patch; rejecting")
            return None

        canaries = sorted(low_cases, key=lambda x: x["accuracy"])[:max_canaries]
        prev_avg = sum([c["accuracy"] for c in canaries]) / len(canaries)
        logger.info(f"Previous canary avg accuracy: {prev_avg:.3f}")

        candidate_full_schema = candidate_prompt.replace("{schema}", self.schema_str)
        canary_accs = []
        for c in canaries:
            full = candidate_full_schema.replace("{raw_text}", c["raw_text"])
            out_raw = self.call_bedrock(full)
            acc = self.calculate_accuracy(c["expected"], out_raw)
            canary_accs.append(acc)
            (self.results_dir / "extractions" / f"patch_cand_{c['file']}_{ts}.json").write_text(json.dumps({"raw": out_raw, "acc": acc}, indent=2), encoding="utf-8")

        cand_avg = sum(canary_accs) / len(canary_accs) if canary_accs else 0.0
        logger.info(f"Candidate patch avg accuracy: {cand_avg:.3f}")

        if cand_avg >= prev_avg + min_delta:
            (self.results_dir / "prompts" / f"patch_accepted_{ts}.txt").write_text(json.dumps(ops, indent=2), encoding="utf-8")
            logger.info("Patch accepted (avg improved).")
            return candidate_prompt
        else:
            (self.results_dir / "prompts" / f"patch_rejected_{ts}.txt").write_text(json.dumps(ops, indent=2), encoding="utf-8")
            logger.info("Patch rejected (no sufficient improvement).")
            return None

    # ---------------- Optimization loop ----------------
    def optimize(self, threshold=0.8, max_iterations=5):
        logger.info("Starting optimization (patch mode)")
        expected_folder = Path("expected-output")
        files = list(expected_folder.glob("*.json"))
        if not files:
            logger.error("No test files in expected-output")
            return self.current_prompt

        logger.info(f"Found {len(files)} test files")

        for it in range(max_iterations):
            logger.info(f"Iteration {it+1}/{max_iterations}")
            low_cases = []
            tested_count = 0
            total_acc = 0.0
            per_field_failures = {}

            for f in files:
                logger.info(f"Processing {f.name}")
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    logger.exception(f"Cannot parse {f}")
                    continue
                raw = data.get("raw_text","")
                expected = data.get("extracted_invoice_values", {})
                if not raw:
                    logger.warning("No raw_text; skipping")
                    continue

                full = self.current_prompt.replace("{schema}", self.schema_str).replace("{raw_text}", raw)
                out_raw = self.call_bedrock(full)
                ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                (self.results_dir / "extractions" / f"extract_{f.stem}_{ts}.json").write_text(json.dumps({"raw": out_raw}, indent=2), encoding="utf-8")

                acc = self.calculate_accuracy(expected, out_raw)
                tested_count += 1
                total_acc += acc

                parsed = self.extract_json_from_text(out_raw) or {}
                parsed_norm = ensure_schema_compliance(parsed, self.schema_template)
                expected_norm = ensure_schema_compliance({"extracted_invoice_values": expected}, self.schema_template)
                field_report = per_field_report(expected_norm, parsed_norm)
                for k, ok in field_report.items():
                    if not ok:
                        per_field_failures[k] = per_field_failures.get(k, 0) + 1

                if acc < threshold:
                    logger.info(f"  ❌ {f.name}: {acc:.3f}")
                    low_cases.append({"file": f.name, "raw_text": raw, "expected": expected, "actual": out_raw, "accuracy": acc})
                else:
                    logger.info(f"  ✅ {f.name}: {acc:.3f}")

                time.sleep(0.4)

            avg = total_acc / tested_count if tested_count else 0.0
            logger.info(f"Iteration summary: avg_accuracy={avg:.3f}, tested={tested_count}")
            logger.info(f"Per-field failure counts: {per_field_failures}")

            if avg >= threshold:
                logger.info("Target reached. Stopping.")
                break

            if not low_cases:
                logger.info("No low cases to optimize further. Stopping.")
                break

            candidate_prompt = self.improve_prompt_with_patch(low_cases)
            if candidate_prompt:
                logger.info("Patch accepted; updating current prompt.")
                self.current_prompt = candidate_prompt
            else:
                logger.info("No acceptable patch; stopping optimization to avoid regressions.")
                break

        Path("optimized_prompt.txt").write_text(self.current_prompt, encoding="utf-8")
        logger.info("Optimization complete. Final prompt saved.")
        logger.info(f"API calls used: {self.api_calls}")
        return self.current_prompt

# ---------------- main ----------------
def main():
    opt = PromptOptimizer()
    opt.optimize(threshold=0.8, max_iterations=5)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Prompt Generator for Meta-Prompt Optimization.
Generates variations of prompt instructions using AI.
"""

import json
from typing import List, Dict, Any
from bedrock_helper import BedrockHelper, extract_json_from_output


class PromptGenerator:
    """
    Generates variations of prompt instructions for optimization.
    """
    
    def __init__(self, region_name: str = 'ap-south-1'):
        self.bedrock_helper = BedrockHelper(region_name=region_name)
        
    def generate_prompt_variations(self, base_prompt: str, variation_strategies: List[str], 
                                 schema: str, domain_context: str = "medical invoice extraction") -> List[str]:
        """
        Generate multiple variations of a base prompt using different strategies.
        
        Args:
            base_prompt: The original prompt to vary
            variation_strategies: List of variation approaches
            schema: JSON schema to maintain
            domain_context: Domain context for variations
            
        Returns:
            List of generated prompt variations
        """
        variations = []
        
        for i, strategy in enumerate(variation_strategies):
            try:
                variation_prompt = f"""You are an expert prompt engineer specializing in {domain_context}. 
Your task is to improve the following prompt using this specific strategy: "{strategy}"

ORIGINAL PROMPT:
{base_prompt}

REQUIREMENTS:
1. Maintain the JSON schema requirement exactly as specified
2. Keep the core functionality for {domain_context}
3. Apply the improvement strategy: {strategy}
4. Output should be more effective than the original
5. Keep medical domain expertise and terminology
6. Ensure the new prompt is clear and actionable

JSON SCHEMA TO MAINTAIN:
{schema}

IMPROVEMENT STRATEGY: {strategy}

Generate an improved version of the prompt that applies this strategy while maintaining all requirements. Return only the improved prompt text, no explanations or markdown formatting."""

                result = self.bedrock_helper.call_bedrock(
                    prompt=variation_prompt,
                    temperature=0.3,  # Some creativity for variations
                    max_tokens=3000,
                    use_cache=True
                )
                
                generated_variation = result['output'].strip()
                
                # Clean up any markdown or extra formatting
                if generated_variation.startswith('```'):
                    lines = generated_variation.split('\n')
                    generated_variation = '\n'.join([line for line in lines if not line.startswith('```')])
                
                variations.append({
                    'strategy': strategy,
                    'prompt_text': generated_variation.strip(),
                    'variation_id': i
                })
                
                print(f"Generated variation {i+1}/{len(variation_strategies)}: {strategy}")
                
            except Exception as e:
                print(f"Failed to generate variation for strategy '{strategy}': {e}")
                continue
        
        return variations
    
    def create_component_based_prompts(self, components: Dict[str, List[str]], 
                                     schema: str) -> List[str]:
        """
        Generate prompts by combining different components.
        
        Args:
            components: Dictionary of component types and options
            schema: JSON schema to include
            
        Returns:
            List of generated prompt combinations
        """
        from itertools import product
        
        # Generate combinations of components
        component_names = list(components.keys())
        component_options = [components[name] for name in component_names]
        
        prompts = []
        
        # Limit combinations to avoid too many (max 20 combinations)
        max_combinations = min(20, len(list(product(*component_options))))
        
        for i, combination in enumerate(product(*component_options)):
            if i >= max_combinations:
                break
                
            # Build prompt from components
            prompt_parts = [
                "<INVOICE_EXTRACTION_SYSTEM>",
                f"<ROLE>\n{combination[component_names.index('role_definition')]}\n</ROLE>",
                "",
                "<EXTRACTION_GUIDELINES>",
                f"{combination[component_names.index('priority_emphasis')]}",
                "",
                f"{combination[component_names.index('error_handling')]}",
                "",
                f"{combination[component_names.index('validation_requirements')]}",
                "</EXTRACTION_GUIDELINES>",
                "",
                f"<SCHEMA_DEFINITION>\n{schema}\n</SCHEMA_DEFINITION>",
                "",
                "<OUTPUT_REQUIREMENTS>",
                "Output ONLY valid JSON matching the schema exactly.",
                "Do NOT include explanations, comments, or markdown formatting.", 
                "Ensure all numeric values are numbers, not strings.",
                "</OUTPUT_REQUIREMENTS>",
                "</INVOICE_EXTRACTION_SYSTEM>"
            ]
            
            prompt_text = "\n".join(prompt_parts)
            prompts.append({
                'combination_id': i,
                'components': dict(zip(component_names, combination)),
                'prompt_text': prompt_text
            })
        
        print(f"Generated {len(prompts)} component-based prompt variations")
        return prompts
    
    def evolve_prompt(self, parent_prompt: str, fitness_score: float, 
                     schema: str, mutation_strength: str = "moderate") -> str:
        """
        Evolve a prompt based on its performance.
        
        Args:
            parent_prompt: The prompt to evolve
            fitness_score: Performance score of the parent (0.0-1.0)
            schema: JSON schema to maintain
            mutation_strength: "light", "moderate", or "aggressive"
            
        Returns:
            Evolved prompt
        """
        mutation_strategies = {
            "light": [
                "Add slight improvements to field extraction clarity",
                "Enhance one specific aspect while keeping everything else the same"
            ],
            "moderate": [
                "Improve instruction clarity and add more specific guidance",
                "Strengthen validation requirements and error prevention",
                "Add domain-specific terminology and context"
            ],
            "aggressive": [
                "Completely restructure for better performance while keeping core requirements",
                "Add comprehensive domain expertise and advanced error handling",
                "Create more sophisticated extraction logic and validation"
            ]
        }
        
        strategy = mutation_strategies[mutation_strength][0] if fitness_score < 0.9 else mutation_strategies[mutation_strength][-1]
        
        evolution_prompt = f"""You are an expert prompt engineer. The following prompt achieved {fitness_score:.1%} accuracy on medical invoice extraction. 

CURRENT PROMPT (Performance: {fitness_score:.1%}):
{parent_prompt}

EVOLUTION STRATEGY: {strategy}

Your task is to evolve this prompt to achieve higher performance. The current performance suggests:
{"- The prompt is working well, make careful improvements" if fitness_score > 0.9 else "- The prompt needs significant improvement" if fitness_score < 0.8 else "- The prompt needs moderate improvement"}

REQUIREMENTS:
1. Maintain the JSON schema exactly
2. Keep medical domain focus  
3. Apply evolution strategy: {strategy}
4. Target performance improvement of 2-5%
5. Keep the prompt practical and actionable

JSON SCHEMA TO MAINTAIN:
{schema}

Return only the evolved prompt text."""

        result = self.bedrock_helper.call_bedrock(
            prompt=evolution_prompt,
            temperature=0.4,  # More creativity for evolution
            max_tokens=3000,
            use_cache=True
        )
        
        return result['output'].strip()


def create_default_prompt_components() -> Dict[str, List[str]]:
    """Create default component options for prompt generation."""
    return {
        "role_definition": [
            "You are an expert medical and pharmacy invoice data extraction AI with extensive training on healthcare billing documents.",
            "You are a specialized AI system designed specifically for accurate extraction of structured data from medical invoices and billing documents.",
            "You are an advanced medical billing AI with deep expertise in healthcare terminology, billing codes, and invoice structures."
        ],
        "priority_emphasis": [
            "CRITICAL REQUIREMENTS:\n1. Extract ALL services/items - Do not skip any service, medication, or billable item\n2. Convert all numeric amounts to numbers (not strings)\n3. Maintain absolute accuracy in medical terminology",
            "PRIMARY OBJECTIVES:\n1. Complete extraction of all billable items and services\n2. Precise numeric data conversion and validation\n3. Perfect preservation of medical context and terminology", 
            "EXTRACTION PRIORITIES:\n1. Comprehensive service identification and extraction\n2. Accurate financial data processing with proper numeric formatting\n3. Medical domain expertise in terminology and classification"
        ],
        "error_handling": [
            "ERROR PREVENTION:\n- Use null for missing string fields, 0 for missing numeric fields\n- Preserve original date formats exactly as found\n- Extract complete facility information including address and contact details",
            "QUALITY ASSURANCE:\n- Apply strict validation to all extracted data\n- Use conservative extraction when uncertain\n- Maintain data integrity throughout the process",
            "DATA VALIDATION:\n- Implement rigorous field validation and type checking\n- Handle missing data with appropriate null/zero values\n- Ensure completeness and accuracy of all extracted information"
        ],
        "validation_requirements": [
            "VALIDATION CHECKS:\n- All amounts must be numeric values, not strings\n- Services array must contain all items found in the invoice\n- Schema compliance is mandatory\n- Medical department identification must be accurate",
            "COMPLIANCE STANDARDS:\n- Strict adherence to JSON schema structure\n- Comprehensive validation of all data types\n- Medical terminology accuracy and consistency\n- Complete extraction without omissions",
            "QUALITY CONTROL:\n- Multi-level validation of extracted data\n- Schema conformance verification\n- Medical domain accuracy assessment\n- Completeness and consistency checks"
        ]
    }


def create_default_variation_strategies() -> List[str]:
    """Create default variation strategies for prompt improvement."""
    return [
        "Add more specific medical terminology and domain expertise guidance",
        "Enhance service extraction instructions with detailed field-by-field rules", 
        "Strengthen error prevention and edge case handling",
        "Improve instruction clarity and reduce ambiguity",
        "Add more comprehensive validation and quality checks",
        "Focus on completeness of extraction to avoid missing any services",
        "Enhance numeric data processing and type conversion instructions",
        "Add more detailed facility and contact information extraction rules"
    ]


if __name__ == "__main__":
    # Test the prompt generator
    generator = PromptGenerator()
    
    # Example base prompt
    base_prompt = """<INVOICE_EXTRACTION_SYSTEM>
<ROLE>
You are an expert medical invoice extraction AI.
</ROLE>

<EXTRACTION_GUIDELINES>
Extract all services and convert amounts to numbers.
Use null for missing fields.
</EXTRACTION_GUIDELINES>

<OUTPUT_REQUIREMENTS>
Output valid JSON only.
</OUTPUT_REQUIREMENTS>
</INVOICE_EXTRACTION_SYSTEM>"""
    
    schema = """{
  "extracted_invoice_values": {
    "invoice_number": "",
    "patient_name": "",
    "services": [{"service": "", "amount": "", "quantity": ""}],
    "total_amount": ""
  }
}"""
    
    print("=== Testing Prompt Generator ===")
    
    # Test variation generation
    strategies = create_default_variation_strategies()[:3]  # Test first 3
    variations = generator.generate_prompt_variations(base_prompt, strategies, schema)
    
    print(f"\nGenerated {len(variations)} prompt variations")
    for i, var in enumerate(variations):
        print(f"\nVariation {i+1} ({var['strategy']}):")
        print(var['prompt_text'][:200] + "...")
    
    # Test component-based generation
    components = create_default_prompt_components()
    component_prompts = generator.create_component_based_prompts(components, schema)
    
    print(f"\nGenerated {len(component_prompts)} component-based prompts")
    
    print("\nPrompt generator test completed!")
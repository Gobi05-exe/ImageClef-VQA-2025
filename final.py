import json
import torch
from PIL import Image
import kagglehub
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import re

def extract_answer(response_text):
    response_text = response_text.strip()
    
    if response_text in ['A', 'B', 'C', 'D', 'E']:
        return response_text

    patterns = [
        r'(?:answer|option|choice)(?:\s+is)?\s*[:\-]?\s*([A-E])\b',
        r'\b([A-E])\b(?:\s+is(?:\s+the)?(?:\s+correct)?\s+(?:answer|option|choice))',
        r'(?:I|the|my)(?:\s+(?:answer|choose|select|pick|opt|go|think))(?:\s+(?:is|for|with))?\s*[:\-]?\s*([A-E])\b',
        r'\b([A-E])\.',
    ]

    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    matches = re.findall(r'\b([A-E])\b', response_text)
    if matches:
        return matches[0].upper()

    for char in response_text:
        if char in ['A', 'B', 'C', 'D', 'E']:
            return char

    return "A"


# Create the BitsAndBytesConfig for 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit quantization
    compute_dtype=torch.bfloat16  # Set the compute dtype (bfloat16 is a good choice for performance)
)

# Load the model with the 4-bit quantization configuration
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-72B-Instruct",
    torch_dtype=torch.bfloat16,  # Use bfloat16 for computation
    device_map="auto",           # Auto-device mapping (CUDA or CPU)
    quantization_config=bnb_config  # Pass the quantization config
)


# Load processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct",use_fast=True)

# Files
INPUT_FILE = "exams_v_test_full_with_paths.json"
OUTPUT_FILE = "run.json"

# Load data
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

results = []

# Process each example
for item in data["rows"]:
    row = item["row"]
    image_path = row["image"]["path"]
    sample_id = row["sample_id"]
    language = row["language"]
    subject = row["subject"]
    grade = row["grade"]

    try:
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Construct prompt
        prompt = (
            f"You are an expert in solving visual multiple-choice exam questions. "
            f"This is an exam question in {language} for Grade {grade} {subject}.\n\n"
            f"Step 1: Carefully extract the question and all answer options (labeled A, B, C, ...), regardless of language.\n"
            f"Step 2: Analyze any diagrams, graphs, tables, or visual content.\n"
            f"Step 3: Reason through the question and choose the best option.\n"
            f"Step 4: Only return the label of the correct option (A, B, C, etc). Do not explain.\n"
            f"Image is provided below.\n"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Prepare model inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        # Inference
        generated_ids = model.generate(**inputs, max_new_tokens=256)
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)
        text = response[0]
        
        # Find the last occurrence of 'assistant' and get everything after it
        last_assistant_index = text.rfind("assistant")
        
        if last_assistant_index != -1:
            # Extract text after 'assistant'
            response = text[last_assistant_index + len("assistant"):].strip()
        else:
            response = text.strip()  # fallback if 'assistant' not found
        
        clean_response = extract_answer(response)

        #print(f"[{sample_id}] → {clean_response}")
        
        print(f"id : {sample_id}\nlanguage : {language}\nanswer_key : {clean_response}\n")

        # Save result
        results.append({
            "id": sample_id,
            "language": language,
            "answer_key": clean_response
        })

        # Free GPU memory
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"❌ Error processing {sample_id}: {e}")

# Save output
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"✅ Saved {len(results)} predictions to {OUTPUT_FILE}")

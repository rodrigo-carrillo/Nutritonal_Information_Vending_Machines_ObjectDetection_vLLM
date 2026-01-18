## Import necessary libraries & set environmental variables
import os
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import requests
import torch
from tqdm import tqdm
import json
import pandas as pd
from pathlib import Path
import re
# os.environ["PYTORCH_METAL_ALLOCATOR_DISABLE_CACHING"] = "1"



## Set HuggingFace cache directory
os.environ['HF_HOME'] = "/Users/rodrigocarrillo/Documents/LLMs Hugging Face/gemma-3-4b-it"
model_id = "google/gemma-3-4b-it"



## Load processor and model
# torch.mps.set_per_process_memory_fraction(0.85)
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="sequential",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    # attn_implementation="eager"
).eval()
processor = AutoProcessor.from_pretrained(model_id)

print(f"✓ Model loaded")
print(f"Model device map: {model.hf_device_map}")
print(f"Model dtype: {model.dtype}")
if hasattr(model, 'hf_device_map'):
    print("Device placement:")
    for name, device in model.hf_device_map.items():
        print(f"  {name}: {device}")



## System prompt
system_prompt = """

You are a helpful assistant.
Your task is to extract the name, size, macronutrients and detailed description of the food products from images.
Use **only** information that is clearly visible in the image.
If the information is not clearly visible, respond with "Not available".
Your response should be in JSON format with the following structure:
{
  "products": [
    {
      "name": "Name or brand of the product.",
      "flavor": "Flavor of the product. It is explicitly said or can be safely inferred from the presentation (e.g., orange package = orange flavor)"
      "size": "Size of the product (e.g., 500ml, 1.0L, 1.0OZ, 50g).",
      "fats": "Amount of fats in grams.",
      "carbohydrates": "Amount of carbohydrates in grams.",
      "proteins": "Amount of proteins in grams.",
      "description": "Detailed description of the product."
    },
    ...
  ]
}
Do not include anything other than the JSON response.

"""



## Run LLM inference on all images in folder
# Configuration
folder_path = "/Users/rodrigocarrillo/Documents/Computer Vision/Vending Machines/03_Cropped_Objects"
output_json_path = "/Users/rodrigocarrillo/Documents/Computer Vision/Vending Machines/04_Output_Data/food_products.json"
output_csv_path = "/Users/rodrigocarrillo/Documents/Computer Vision/Vending Machines/04_Output_Data/food_products.csv"

# Process all JPG files
results = []

# Get all .jpg files (case-insensitive)
jpg_files = list(Path(folder_path).glob("*.jpg")) + list(Path(folder_path).glob("*.JPG"))
print(f"Found {len(jpg_files)} JPG files to process\n")

for idx, image_path in tqdm(enumerate(jpg_files, 1), total=len(jpg_files), desc="Processing images"):
    print(f"Processing {idx}/{len(jpg_files)}: {image_path.name}")
    
    try:
        # Prepare messages for this image
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": "Get the name and detailed description of the food product from this image."}
                ]
            }
        ]
        
        # Process with model
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)
        
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            generation = generation[0][input_len:]
        
        decoded = processor.decode(generation, skip_special_tokens=True)
        print(decoded)
        
        # Clean up the decoded string (remove markdown code blocks if present)
        decoded_clean = decoded.strip()
        if decoded_clean.startswith("```json"):
            decoded_clean = decoded_clean.replace("```json", "").replace("```", "").strip()
        
        # Parse the JSON response
        try:
            result_dict = json.loads(decoded_clean)
            
            # Extract the first product from the products array
            if "products" in result_dict and len(result_dict["products"]) > 0:
                product = result_dict["products"][0]
                
                result_entry = {
                    "filename": image_path.name,
                    "name": product.get("name", "Unknown"),
                    "flavor": product.get("flavor", "Unknown"),
                    "description": product.get("description", ""),
                    "size": product.get("size", ""),
                    "fats": product.get("fats", ""),
                    "carbohydrates": product.get("carbohydrates", ""),
                    "proteins": product.get("proteins", "")
                }
            else:
                # Fallback if structure is unexpected
                result_entry = {
                    "filename": image_path.name,
                    "name": "Unknown",
                    "description": decoded_clean
                }
                
        except json.JSONDecodeError as je:
            print(f"  ⚠ JSON parsing error: {je}")
            result_entry = {
                "filename": image_path.name,
                "name": "Parse Error",
                "description": decoded_clean
            }
        
        results.append(result_entry)
        print(f"  ✓ {result_entry.get('name', 'Unknown')}")
        
    except Exception as e:
        print(f"  ✗ Error processing {image_path.name}: {e}")
        results.append({
            "filename": image_path.name,
            "name": "Error",
            "description": str(e)
        })

print(f"\n{'='*50}")
print(f"Processing complete!")
print(f"{'='*50}\n")



## Save as JSON
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✓ JSON saved to: {output_json_path}")

# Create DataFrame and save as CSV
df = pd.DataFrame(results)

# Reorder columns - put filename and name first, then description
columns_order = ['filename', 'name', 'description']
# Add any additional columns that exist
additional_cols = [col for col in df.columns if col not in columns_order]
columns_order.extend(additional_cols)

df = df[columns_order]
df.to_csv(output_csv_path, index=False, encoding='utf-8')

print(f"✓ CSV saved to: {output_csv_path}")
print(f"\nProcessed {len(results)} images successfully!")

# Display summary
print(f"\n{'='*50}")
print("Preview of results:")
print(f"{'='*50}")
print(df.head(10).to_string(index=False))



## Clean memory
del model
del processor
import gc
gc.collect()

if torch.backends.mps.is_available():
    torch.mps.empty_cache()

print("✓ Memory cleared")
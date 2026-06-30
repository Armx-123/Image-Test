import os
import torch
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForCausalLM

# Bypass the flash_attn hardware bug
def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

model_id = "microsoft/Florence-2-base"
print("Loading Florence-2 model for OCR onto CPU...")

with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to("cpu")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

def run_ocr(image_path, output_path):
    image = Image.open(image_path).convert("RGB")
    
    # 1. Command the model to find text and its coordinates
    task_prompt = "<OCR_WITH_REGION>"
    
    inputs = processor(text=task_prompt, images=image, return_tensors="pt").to("cpu")
    
    print(f"Scanning for text in {image_path}...")
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=2048, # Increased token limit in case there is a lot of text
            num_beams=3
        )
    
    # 2. Decode while preserving the location tokens
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )
    
    detections = parsed_answer.get(task_prompt, {})
    
    # OCR returns "quad_boxes" (8 coordinates for a polygon) instead of standard 4-point boxes
    quad_boxes = detections.get('quad_boxes', [])
    labels = detections.get('labels', [])
    
    print(f"Model found {len(labels)} distinct text regions. Highlighting...")
    
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except IOError:
        font = None

    for box, label in zip(quad_boxes, labels):
        # Draw a yellow polygon around the text
        draw.polygon(box, outline="yellow", width=2)
        
        # Grab the top-left coordinate to place the label
        x1, y1 = box[0], box[1]
        draw.rectangle([x1, y1 - 15, x1 + (len(label) * 6), y1], fill="yellow")
        draw.text((x1 + 2, y1 - 13), label, fill="black", font=font)
        
    image.save(output_path)
    print(f"Success! OCR Output saved to {output_path}")

if __name__ == "__main__":
    input_img = "input.jpg"
    output_img = "output_ocr.jpg"
    
    if os.path.exists(input_img):
        run_ocr(input_img, output_img)
    else:
        print(f"Error: {input_img} not found in the repository root.")

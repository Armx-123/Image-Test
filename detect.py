import os
import torch
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForCausalLM

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

model_id = "microsoft/Florence-2-base"
print("Loading Florence-2 model onto CPU...")

with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to("cpu")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

def run_detection(image_path, output_path, user_prompt):
    # Keep the original image pristine for cropping
    original_image = Image.open(image_path).convert("RGB")
    
    # Create a copy specifically for drawing the red boxes
    highlight_image = original_image.copy()
    
    task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
    full_prompt = task_prompt + user_prompt
    
    inputs = processor(text=full_prompt, images=original_image, return_tensors="pt").to("cpu")
    
    print(f"Searching for '{user_prompt}' in {image_path}...")
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(original_image.width, original_image.height)
    )
    
    detections = parsed_answer[task_prompt]
    num_detections = len(detections.get('bboxes', []))
    print(f"Model successfully found {num_detections} matching objects. Processing...")
    
    draw = ImageDraw.Draw(highlight_image)
    try:
        font = ImageFont.load_default()
    except IOError:
        font = None

    largest_box = None
    largest_area = -1

    for box, label in zip(detections['bboxes'], detections['labels']):
        x1, y1, x2, y2 = box
        
        # 1. Calculate the area of the current box
        area = (x2 - x1) * (y2 - y1)
        if area > largest_area:
            largest_area = area
            largest_box = (x1, y1, x2, y2)

        # 2. Draw on the copy, NOT the original
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.rectangle([x1, y1 - 15, x1 + (len(label) * 6), y1], fill="red")
        draw.text((x1 + 2, y1 - 13), label, fill="white", font=font)
        
    highlight_image.save(output_path)
    print(f"Success! Output saved to {output_path}")

    # 3. Crop out the largest object from the pristine original image
    if largest_box:
        cropped_largest = original_image.crop(largest_box)
        cropped_largest.save("largest_object.jpg")
        print(f"Largest object isolated and saved as largest_object.jpg (Area: {largest_area:.1f}px)")

if __name__ == "__main__":
    input_img = "inpu.png"
    output_img = "output_detected.jpg"
    prompt_file = "prompt.txt"
    
    if os.path.exists(prompt_file):
        with open(prompt_file, "r") as f:
            user_text = f.read().strip()
    else:
        user_text = "a Kreo Hive 65 mechanical gaming keyboard"
        
    if os.path.exists(input_img):
        run_detection(input_img, output_img, user_text)
    else:
        print(f"Error: {input_img} not found in the repository root.")

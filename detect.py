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
    image = Image.open(image_path).convert("RGB")
    
    # 1. The new task for text-prompted detection
    task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
    
    # 2. Combine the task instruction with your custom text
    full_prompt = task_prompt + user_prompt
    
    inputs = processor(text=full_prompt, images=image, return_tensors="pt").to("cpu")
    
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
        image_size=(image.width, image.height)
    )
    
    detections = parsed_answer[task_prompt]
    
    num_detections = len(detections.get('bboxes', []))
    print(f"Model successfully found {num_detections} matching objects. Drawing boxes...")
    
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except IOError:
        font = None

    for box, label in zip(detections['bboxes'], detections['labels']):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.rectangle([x1, y1 - 15, x1 + (len(label) * 6), y1], fill="red")
        draw.text((x1 + 2, y1 - 13), label, fill="white", font=font)
        
    image.save(output_path)
    print(f"Success! Output saved to {output_path}")

if __name__ == "__main__":
    input_img = "input.jpg"
    output_img = "output_detected.jpg"
    prompt_file = "prompt.txt"
    
    # 3. Read the custom prompt from a text file
    if os.path.exists(prompt_file):
        with open(prompt_file, "r") as f:
            user_text = f.read().strip()
    else:
        # Fallback if you forget to include the text file
        user_text = "a Kreo Hive 65 mechanical gaming keyboard"
        
    if os.path.exists(input_img):
        run_detection(input_img, output_img, user_text)
    else:
        print(f"Error: {input_img} not found in the repository root.")

import os
import torch
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForCausalLM

# 1. Workaround to intercept Hugging Face's erroneous flash_attn requirement
def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

model_id = "microsoft/Florence-2-base"
print("Loading Florence-2 model onto CPU...")

# 2. Apply the patch while initializing the Model and Processor
with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to("cpu")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

def run_detection(image_path, output_path):
    image = Image.open(image_path).convert("RGB")
    
    task_prompt = "<DENSE_REGION_CAPTION>"
    inputs = processor(text=task_prompt, images=image, return_tensors="pt").to("cpu")
    
    print(f"Processing {image_path}...")
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )
    
    # 3. THE FIX: Set skip_special_tokens to False to preserve the bounding box coordinates
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )
    
    detections = parsed_answer[task_prompt]
    
    # Debug logging to confirm detections in GitHub Actions
    num_detections = len(detections.get('bboxes', []))
    print(f"Model successfully found {num_detections} objects. Drawing boxes...")
    
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
    
    if os.path.exists(input_img):
        run_detection(input_img, output_img)
    else:
        print(f"Error: {input_img} not found in the repository root.")

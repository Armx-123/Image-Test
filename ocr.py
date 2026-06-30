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
    original_image = Image.open(image_path).convert("RGB")
    # Create a copy for drawing highlights so the crop sources remain clean
    highlight_image = original_image.copy()
    
    task_prompt = "<OCR_WITH_REGION>"
    inputs = processor(text=task_prompt, images=original_image, return_tensors="pt").to("cpu")
    
    print(f"Scanning for text in {image_path}...")
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=2048,
            num_beams=3
        )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(original_image.width, original_image.height)
    )
    
    detections = parsed_answer.get(task_prompt, {})
    quad_boxes = detections.get('quad_boxes', [])
    labels = detections.get('labels', [])
    
    if not quad_boxes:
        print("No text regions detected in the image.")
        return

    print(f"Model found {len(labels)} distinct text regions. Processing areas...")
    
    draw = ImageDraw.Draw(highlight_image)
    try:
        font = ImageFont.load_default()
    except IOError:
        font = None

    largest_box = None
    largest_area = -1
    smallest_box = None
    smallest_area = float('inf')

    for box, label in zip(quad_boxes, labels):
        # Draw a yellow polygon around the text on the highlighted canvas
        draw.polygon(box, outline="yellow", width=2)
        x1, y1 = box[0], box[1]
        draw.rectangle([x1, y1 - 15, x1 + (len(label) * 6), y1], fill="yellow")
        draw.text((x1 + 2, y1 - 13), label, fill="black", font=font)
        
        # Calculate bounding box bounds from the 8-point polygon coords [x1,y1,x2,y2,x3,y3,x4,y4]
        x_coords = box[0::2]
        y_coords = box[1::2]
        xmin, xmax = min(x_coords), max(x_coords)
        ymin, ymax = min(y_coords), max(y_coords)
        
        # Calculate area
        area = (xmax - xmin) * (ymax - ymin)
        
        # Skip invalid or 0-sized bounding regions
        if area <= 0:
            continue
            
        # Keep track of largest and smallest boxes
        if area > largest_area:
            largest_area = area
            largest_box = (xmin, ymin, xmax, ymax)
            
        if area < smallest_area:
            smallest_area = area
            smallest_box = (xmin, ymin, xmax, ymax)

    # Save the full highlighted image
    highlight_image.save(output_path)
    print(f"Main OCR Output saved to {output_path}")

    # Crop and save the largest text box image if found
    if largest_box:
        cropped_largest = original_image.crop(largest_box)
        cropped_largest.save("largest_text.jpg")
        print(f"Largest text box cropped and saved (Area: {largest_area:.1f}px)")

    # Crop and save the smallest text box image if found
    if smallest_box:
        cropped_smallest = original_image.crop(smallest_box)
        cropped_smallest.save("smallest_text.jpg")
        print(f"Smallest text box cropped and saved (Area: {smallest_area:.1f}px)")

if __name__ == "__main__":
    input_img = "input.jpg"
    output_img = "output_ocr.jpg"
    
    if os.path.exists(input_img):
        run_ocr(input_img, output_img)
    else:
        print(f"Error: {input_img} not found in the repository root.")

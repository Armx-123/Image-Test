import os
import json
import torch
import cv2
import numpy as np
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

def rgb_to_hex(r, g, b):
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"

def get_top_edge_colors(cv_rgb_img):
    height, width = cv_rgb_img.shape[:2]
    top_y = 0
    
    # PIL image arrays are already RGB, no BGR-flipping required
    left_rgb = cv_rgb_img[top_y, 0]
    middle_rgb = cv_rgb_img[top_y, width // 2]
    right_rgb = cv_rgb_img[top_y, width - 1]
    
    return {
        "top_left": rgb_to_hex(*left_rgb),
        "top_middle": rgb_to_hex(*middle_rgb),
        "top_right": rgb_to_hex(*right_rgb)
    }

def get_text_color(cv_rgb_crop):
    gray = cv2.cvtColor(cv_rgb_crop, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    white_pixels = cv2.countNonZero(thresh)
    black_pixels = thresh.size - white_pixels
    
    # Text is always the mathematical minority of the bounding box area
    if white_pixels < black_pixels:
        text_mask = thresh == 255
    else:
        text_mask = thresh == 0
        
    pixels = cv_rgb_crop[text_mask]
    
    if len(pixels) == 0:
        return None
        
    mean_rgb = np.mean(pixels, axis=0)
    return rgb_to_hex(mean_rgb[0], mean_rgb[1], mean_rgb[2])

model_id = "microsoft/Florence-2-base"
print("Loading Florence-2 model for OCR onto CPU...")

with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to("cpu")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

def run_ocr(image_path):
    original_image = Image.open(image_path).convert("RGB")
    cv_img = np.array(original_image)
    
    # Extract edge colors immediately
    edge_colors = get_top_edge_colors(cv_img)
    
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

    largest_data = None
    largest_area = -1
    smallest_data = None
    smallest_area = float('inf')

    for box, label in zip(quad_boxes, labels):
        x_coords = box[0::2]
        y_coords = box[1::2]
        xmin, xmax = int(min(x_coords)), int(max(x_coords))
        ymin, ymax = int(min(y_coords)), int(max(y_coords))
        
        area = (xmax - xmin) * (ymax - ymin)
        
        if area <= 0:
            continue
            
        box_data = {
            "text": label,
            "box": (xmin, xmax, ymin, ymax)
        }
            
        if area > largest_area:
            largest_area = area
            largest_data = box_data
            
        if area < smallest_area:
            smallest_area = area
            smallest_data = box_data

    output = {}

    for name, entry in [("largest", largest_data), ("smallest", smallest_data)]:
        if not entry:
            continue
            
        xmin, xmax, ymin, ymax = entry["box"]
        crop = cv_img[ymin:ymax, xmin:xmax]
        color = get_text_color(crop)

        output[name] = {
            "text": entry["text"],
            "color": color,
            "bounding_box": {
                "xmin": xmin,
                "xmax": xmax,
                "ymin": ymin,
                "ymax": ymax
            }
        }

    output["poster_edge_colors"] = edge_colors

    with open("text_colors.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)
        
    print("Success! JSON data extracted and saved to text_colors.json:")
    print(json.dumps(output, indent=4))

if __name__ == "__main__":
    input_img = "inpu.png"
    
    if os.path.exists(input_img):
        run_ocr(input_img)
    else:
        print(f"Error: {input_img} not found in the repository root.")

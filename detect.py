import os
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForCausalLM

# 1. Initialize Model and Processor (Forces CPU execution)
model_id = "microsoft/Florence-2-base"
print("Loading Florence-2 model onto CPU...")
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to("cpu")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

def run_detection(image_path, output_path):
    image = Image.open(image_path).convert("RGB")
    
    # 2. Prepare the prompt for automatic out-of-the-box dense labeling
    task_prompt = "<DENSE_REGION_CAPTION>"
    inputs = processor(text=task_prompt, images=image, return_tensors="pt").to("cpu")
    
    # 3. Generate predictions
    print(self_generated_text := f"Processing {image_path}...")
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )
    
    # 4. Parse answers back to pixel coordinates
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )
    
    # 5. Draw bounding boxes and labels onto the image
    detections = parsed_answer[task_prompt]
    draw = ImageDraw.Draw(image)
    
    # Try using a default font, fallback if unavailable
    try:
        font = ImageFont.load_default()
    except IOError:
        font = None

    for box, label in zip(detections['bboxes'], detections['labels']):
        x1, y1, x2, y2 = box
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        # Draw text label background
        draw.rectangle([x1, y1 - 15, x1 + (len(label) * 6), y1], fill="red")
        # Draw text
        draw.text((x1 + 2, y1 - 13), label, fill="white", font=font)
        
    image.save(output_path)
    print(f"Success! Output saved to {output_path}")

if __name__ == "__main__":
    # Expects an image named 'input.jpg' in the repo root
    input_img = "input.jpg"
    output_img = "output_detected.jpg"
    
    if os.path.exists(input_img):
        run_detection(input_img, output_img)
    else:
        print(f"Error: {input_img} not found in the repository root.")

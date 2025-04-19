import os
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
import torch

# --- CONFIGURATION ---
INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "outpainted_images"
MODEL_ID = "stabilityai/stable-diffusion-2-inpainting"
#MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"
PROMPT = "a realistic image of a car, the background seamlessly extends to match the original scene, matching the original background."
NEGATIVE_PROMPT = "new objects, text, abrupt color changes, mismatched background, blurry, distorted, extra objects, watermark, frame, border, signature"
GUIDANCE_SCALE = 6
NUM_INFERENCE_STEPS = 150

# --- SETUP ---
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe = pipe.to(device)

def make_square_canvas(img, fill=(255, 255, 255)):
    w, h = img.size
    size = max(w, h)
    new_img = Image.new("RGB", (size, size), fill)
    x = (size - w) // 2
    y = (size - h) // 2
    new_img.paste(img, (x, y))
    return new_img, (x, y, x + w, y + h)

def make_mask(img, box):
    # Mask is white (255) everywhere (outpaint), black (0) over original image (preserve)
    mask = Image.new("L", img.size, 255)
    mask.paste(0, box)
    return mask

# --- PROCESSING ---
for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        print(f"Processing {filename}...")
        img_path = os.path.join(INPUT_FOLDER, filename)
        img = Image.open(img_path).convert("RGB")
        square_img, box = make_square_canvas(img)
        mask = make_mask(square_img, box)
        result = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            image=square_img,
            mask_image=mask,
            guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=NUM_INFERENCE_STEPS,
        ).images[0]
        result.save(os.path.join(OUTPUT_FOLDER, filename))
        print(f"Saved: {os.path.join(OUTPUT_FOLDER, filename)}")

print("Batch outpainting complete!")

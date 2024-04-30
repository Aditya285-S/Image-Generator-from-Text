from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import torch

model_id1 = "dreamlike-art/dreamlike-diffusion-1.0"

pipe = StableDiffusionPipeline.from_pretrained(model_id1, use_safetensors=True)
pipe = pipe.to("cpu")

def generate_image_interface(prompt):
    # Assuming `pipe` is correctly defined elsewhere for image generation
    params = {
        'prompt': prompt,
        'num_inference_steps': 100,
        'num_images_per_prompt': 2,  # Assuming this is a valid parameter
        'height': int(1.2 * 640)  # Assuming height is calculated based on weight
    }

    # Assuming `pipe` is correctly defined elsewhere
    img = pipe(**params).images
    return img[0], img[1]

import gradio as gr

demo = gr.Interface(
    fn=generate_image_interface,
    inputs=["text"],
    outputs=["image", "image"],
    title="Image Generation Interface",
    description="Generate images based on prompts."
)

demo.launch()

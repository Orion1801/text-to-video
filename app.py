import gradio as gr
import torch
import os
import spaces
import uuid

from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_video
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from PIL import Image

# Constants
bases = {
    "ToonYou": "frankjoshua/toonyou_beta6",
    "epiCRealism": "emilianJR/epiCRealism"
}
step_loaded = None
base_loaded = "ToonYou"
motion_loaded = None

# Ensure model and scheduler are initialized in GPU-enabled function
if not torch.cuda.is_available():
    raise NotImplementedError("No GPU detected!")

device = "cuda"
dtype = torch.float16
pipe = AnimateDiffPipeline.from_pretrained(bases[base_loaded], torch_dtype=dtype).to(device)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")

# Safety checkers
from safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor

safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker").to(device)
feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")

def check_nsfw_images(images: list[Image.Image]) -> list[bool]:
    safety_checker_input = feature_extractor(images, return_tensors="pt").to(device)
    has_nsfw_concepts = safety_checker(images=[images], clip_input=safety_checker_input.pixel_values.to(device))
    return has_nsfw_concepts

# Function 
@spaces.GPU(enable_queue=True)
def generate_image(prompt, base, motion, step, progress=gr.Progress()):
    global step_loaded
    global base_loaded
    global motion_loaded
    print(prompt, base, step)

    if step_loaded != step:
        repo = "ByteDance/AnimateDiff-Lightning"
        ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
        pipe.unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device), strict=False)
        step_loaded = step

    if base_loaded != base:
        pipe.unet.load_state_dict(torch.load(hf_hub_download(bases[base], "unet/diffusion_pytorch_model.bin"), map_location=device), strict=False)
        base_loaded = base

    if motion_loaded != motion:
        pipe.unload_lora_weights()
        if motion != "":
            pipe.load_lora_weights(motion, adapter_name="motion")
            pipe.set_adapters(["motion"], [0.7])
        motion_loaded = motion

    progress((0, step))
    def progress_callback(i, t, z):
        progress((i+1, step))

    output = pipe(prompt=prompt, guidance_scale=1.0, num_inference_steps=step, callback=progress_callback, callback_steps=1)

    has_nsfw_concepts = check_nsfw_images([output.frames[0][0]])
    if has_nsfw_concepts[0]:
        gr.Warning("NSFW content detected.")
        return None

    name = str(uuid.uuid4()).replace("-", "")
    path = f"/tmp/{name}.mp4"
    export_to_video(output.frames[0], path, fps=10)
    return path


# Gradio Interface
with gr.Blocks(css="style.css") as demo:
    gr.HTML(
        "<h1><center>  Arcane® AI ⚡</center></h1>")
    with gr.Group():
        with gr.Row():
            prompt = gr.Textbox(
                label='Prompt (English)'
            )
        with gr.Row():
            select_base = gr.Dropdown(
                label='Base model',
                choices=[
                    "ToonYou",
                    "epiCRealism"
                ],
                value=base_loaded,
                interactive=True
            )
            select_motion = gr.Dropdown(
                label='Motion',
                choices=[
                    ("Default", ""),
                    ("Zoom in", "guoyww/animatediff-motion-lora-zoom-in"),
                    ("Zoom out", "guoyww/animatediff-motion-lora-zoom-out"),
                    ("Tilt up", "guoyww/animatediff-motion-lora-tilt-up"),
                    ("Tilt down", "guoyww/animatediff-motion-lora-tilt-down"),
                    ("Pan left", "guoyww/animatediff-motion-lora-pan-left"),
                    ("Pan right", "guoyww/animatediff-motion-lora-pan-right"),
                    ("Roll left", "guoyww/animatediff-motion-lora-rolling-anticlockwise"),
                    ("Roll right", "guoyww/animatediff-motion-lora-rolling-clockwise"),
                ],
                value="",
                interactive=True
            )
            select_step = gr.Dropdown(
                label='Inference steps',
                choices=[
                    ('1-Step', 1), 
                    ('2-Step', 2),
                    ('4-Step', 4),
                    ('8-Step', 8)],
                value=8,
                interactive=True
            )
            submit = gr.Button(
                scale=1,
                variant='primary'
            )
    video = gr.Video(
        label='AnimateDiff-Lightning',
        autoplay=True,
        height=512,
        width=512,
        elem_id="video_output"
    )

    prompt.submit(
        fn=generate_image,
        inputs=[prompt, select_base, select_motion, select_step],
        outputs=video,
    )
    submit.click(
        fn=generate_image,
        inputs=[prompt, select_base, select_motion, select_step],
        outputs=video,
    )

demo.queue().launch(share=True)
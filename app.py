from sldl.video import VideoSR
from sldl.image import ImageSR

import gradio as gr
import tempfile
import shutil
import torch
import ffmpeg
import time
from PIL import Image

cc = 2
if torch.backends.mps.is_available():
    device = 'mps'
    cc = 5
elif torch.cuda.is_available():
    device = 'cuda'
    cc = 10
else:
    device = 'auto'

vbsrgan = VideoSR('BSRGAN').to(device)
vresrgan = VideoSR('RealESRGAN').to(device)
ibsrgan = ImageSR('BSRGAN').to(device)
iresrgan = ImageSR('RealESRGAN').to(device)

def upscale_video(input_video, output_video, progress, mname):
    modelname = mname.lower()
    model = vbsrgan
    if modelname == 'bsrgan (default)':
        # do nothing
        pass
    elif modelname == 'real esrgan':
        model = vresrgan
    model(input_video, output_video, progress.tqdm)

def upscale_image(input_image, output_image, mname):
    modelname = mname.lower()
    model = ibsrgan
    if modelname == 'bsrgan (default)':
        # do nothing
        pass
    elif modelname == 'real esrgan':
        model = oresrgan
    model(input_image, output_image)

# Gradio interface
def video_upscaling_interface(input_text, model_name, progress=gr.Progress()):
    if input_text:
        temp_dir = tempfile.mkdtemp()
        input_video_path = f"{temp_dir}/input_video"
        output_video_path = f"{temp_dir}/output_video.mp4"

        # Convert input video to mp4 using ffmpeg
        ffmpeg.input(input_text).output(input_video_path + '.mp4').run()

        # Upscale the video
        upscale_video(input_video_path + '.mp4', output_video_path, progress, model_name)

        return [output_video_path, output_video_path]
    else:
        return ["no_vid.mp4", "no_vid.mp4"]


def image_upscaling_interface(input_text, model_name):
    if input_text:
        temp_dir = tempfile.mkdtemp()
        input_image_path = f"{temp_dir}/input_image.jpg"
        output_image_path = f"{temp_dir}/output_image.jpg"
        image = Image.open(input_text)
        image.save(input_image_path)
        upscale_image(input_image_path, output_image_path, model_name)
        return [output_image_path, output_image_path]
    else:
        return ["no_image.jpg", "no_image.jpg"]
    

css = "footer {display: none !important;} .gradio-container {min-height: 0px !important;}"


with gr.Blocks(css=css) as demo:
    gr.Markdown("# Upscale by CVSYS")
    with gr.Tab("Image"):
        with gr.Row():
            with gr.Column():
                iinp = gr.Image(label="Upload Image", interactive=True)
                imod = gr.Dropdown(
                    ["BSRGAN (Default)", "Real ESRGAN"],
                    value="BSRGAN (Default)",
                    interactive=True,
                    label="Model"
                )
            with gr.Column():
                iout = gr.Image(label="View Image", interactive=False)
                ifile = gr.File(label="Download Image", interactive=False)
        ibtn = gr.Button(value="Upscale Image")
    with gr.Tab("Video"):
        with gr.Row():
            with gr.Column():
                vinp = gr.Video(label="Upload Video", interactive=True)
                vmod = gr.Dropdown(
                    ["BSRGAN (Default)", "Real ESRGAN"],
                    value="BSRGAN (Default)",
                    interactive=True,
                    label="Model"
                )
            with gr.Column():
                vout = gr.Video(label="Watch Video", interactive=False)
                vfile = gr.File(label="Download Video", interactive=False)
        vbtn = gr.Button(value="Upscale Video")
    ibtn.click(image_upscaling_interface, [iinp, imod], [iout, ifile])
    vbtn.click(video_upscaling_interface, [vinp, vmod], [vout, vfile])
    demo.queue(concurrency_count=cc)
    demo.launch()

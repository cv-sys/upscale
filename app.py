from sldl.video import VideoSR

import gradio as gr
import tempfile
import shutil
import torch
import time
cc = 2
if torch.backends.mps.is_available():
    device = 'mps'
    cc = 5
elif torch.cuda.is_available():
    device = 'cuda'
    cc = 10
else:
    device = 'auto'

bsrgan = VideoSR('BSRGAN').to(device)
resrgan = VideoSR('RealESRGAN').to(device)

def upscale_video(input_video, output_video, progress, mname):
    print(f'Upscaling video. Input: {input_video}')
    modelname = mname.lower()
    model = bsrgan
    if modelname == 'bsrgan (default)':
        # do nothing
        pass
    elif modelname == 'real esrgan':
        model = resrgan
    model(input_video, output_video, progress.tqdm)
    # imgs = [None] * 24
    # for img in progress.tqdm(imgs, desc="Loading..."):
        # time.sleep(0.1)
    # shutil.copy(input_video, output_video)

# Gradio interface
def video_upscaling_interface(input_text, model_name, progress=gr.Progress()):
    if input_text:
        temp_dir = tempfile.mkdtemp()
        input_video_path = f"{temp_dir}/input_video.mp4"
        shutil.copy(input_text, input_video_path)
        output_video_path = f"{temp_dir}/output_video.mp4"
        upscale_video(input_video_path, output_video_path, progress, model_name)
        return [output_video_path, output_video_path]
    else:
        return ["no_vid.mp4", "no_vid.mp4"]

css = "footer {display: none !important;} .gradio-container {min-height: 0px !important;}"



input_interfaces = [
    gr.inputs.Video(label="Upload Video"),
    gr.Dropdown(
        ["BSRGAN (Default)", "Real ESRGAN"],
        label="Model",
        value="BSRGAN"
    )
]
output_interfaces = [
    gr.outputs.Video(label="Watch Video"),
    gr.outputs.File(label="Download Video"),
]

# Create the Gradio interface
app = gr.Interface(video_upscaling_interface, input_interfaces, css=css, outputs=output_interfaces)

app.queue(concurrency_count=cc)
app.launch()
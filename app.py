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
    modelname = mname.lower()
    model = bsrgan
    if modelname == 'bsrgan (default)':
        # do nothing
        pass
    elif modelname == 'real esrgan':
        model = resrgan
    model(input_video, output_video, progress.tqdm)

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


with gr.Blocks(css=css) as demo:
    gr.Markdown("# Upscale by CVSYS")
    with gr.Row():
        with gr.Column():
            vid = gr.Video(label="Upload Video", interactive=True)
            mod = gr.Dropdown(
                ["BSRGAN (Default)", "Real ESRGAN"],
                value="BSRGAN (Default)",
                interactive=True,
                label="Model"
            )
        with gr.Column():
            vidout = gr.Video(label="Watch Video", interactive=False)
            file = gr.File(label="Download Video", interactive=False)
    btn = gr.Button(value="Upscale")
    btn.click(video_upscaling_interface, [vid, mod], [vidout, file])
    demo.queue(concurrency_count=cc)
    demo.launch()

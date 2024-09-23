import os

# set TMPDIR to avoid permission problem
current_directory = os.getcwd()
os.environ['TMPDIR'] = os.path.join(current_directory,'.cache')

import torch
import gradio as gr
from demo_utils import *
seed=42
set_seed(seed)
args = parse_args()
torch_device = "cuda" if torch.cuda.is_available() else "cpu"


HEIGHT, WIDTH = args.resolution

if args.ckpt_path:
    repo_id = args.ckpt_path
else:
    repo_id = find_latest_checkpoint()
ckpt_name = repo_id.split('/')[-1]
print(f'Load ckpt: {ckpt_name}')
if args.debug:
    model = None
    processor = {
        'image_processer': None,
        'diffusion_image_processor': None,
        'tokenizer': None
    }
else:
    model, processor = load_wm(repo_id =repo_id)
chatwm = ChatWM(model,processor)

os.makedirs('video_output', exist_ok=True)

# title = """<h1 align="center"><a href="https://maitrix.org/"><img src="https://maitrix.org/assets/img/logo-title.png" alt="World Model" border="0" style="margin: 0 auto; height: 50px;" /></a> </h1>"""
description = (
    """<br><a href='http://71.142.245.226:8583/'>
    # Pandora
    <img src='https://img.shields.io/badge/Github-Code-blue'></a><p>
    - Upload An Image
    - Press Generate
    """
)

demo = gr.Blocks(theme=gr.themes.Soft(primary_hue="slate",))
with demo:
    # gr.Markdown(title)
    gr.Markdown(description)
    if args.debug:
        gr.Markdown("***Debug Mode, No Model loaded***")

    gr.Markdown(f"Current checkpoint: {ckpt_name}")
    with gr.Tabs():
        with gr.Row():
            with gr.Column(visible=True, scale=0.6)  as input_raws:
                image_input = gr.Image(label='Current State',height=HEIGHT,width=1024)
                text_input = gr.Textbox(label='Text Control Action')
                with gr.Row():
                    round1_button = gr.Button("💭 Action 1",visible=True, interactive=True,variant="primary")
                    round2_button = gr.Button("💭 Action 2",visible=True, interactive=False,variant="primary")
                    round3_button = gr.Button("💭 Action 3",visible=True, interactive=False,variant="primary")
                with gr.Row():
                    round4_button = gr.Button("💭 Action 4",visible=True, interactive=False,variant="primary")
                    round5_button = gr.Button("💭 Action 5",visible=True, interactive=False,variant="primary")
                    multi_button = gr.Button("💭 Multi-Action",visible=True, interactive=True,variant="primary")

                with gr.Row():
                    clear_button = gr.Button("Clear",visible=True, interactive=True)
                gr.Markdown(" ")
            with gr.Column(visible=True, scale=0.4)  as input_raws:
                fs, n_samples, unconditional_guidance_scale, ddim_steps, ddim_eta, num_round, random_seed = init_sliders(seed)
                random_seed.submit(fn=set_seed, inputs=[random_seed], outputs=[random_seed])
                gr.Markdown(" ")
        with gr.Row():
            video_output_0 = gr.Video(width=WIDTH,height=HEIGHT,label='Final Output')
            video_output_1 = gr.Video(width=WIDTH,height=HEIGHT, label='Action 1')
            video_output_2 = gr.Video(width=WIDTH,height=HEIGHT, label='Action 2')
        with gr.Row():
            video_output_3 = gr.Video(width=WIDTH,height=HEIGHT, label='Action 3')
            video_output_4 = gr.Video(width=WIDTH,height=HEIGHT, label='Action 4')
            video_output_5 = gr.Video(width=WIDTH,height=HEIGHT, label='Action 5')
        with gr.Row():
            examples = gr.Examples(
                examples=[
                    ['examples/car.png', 'The car moves forward.'],
                    ['examples/bench.png', 'Wind flows the leaves.'],
                    ['examples/mountain.png', 'The sky gets dark.'],
                    # ['examples/red_car.png', 'Explosion happens.'],
                ],
                inputs=[image_input, text_input, ddim_steps, fs, 
                        n_samples,unconditional_guidance_scale, ddim_eta]
            )

    video_output = [video_output_0, video_output_1, video_output_2, video_output_3, video_output_4, video_output_5]
    button_output = [round1_button,round2_button,round3_button,round4_button,round5_button, multi_button]
    text_image_output = [image_input, text_input]
    total_output = button_output + video_output + text_image_output

    round1_button.click(chatwm.generate_video, inputs=[image_input, text_input, ddim_steps, fs, 
                                                      n_samples,unconditional_guidance_scale, ddim_eta,random_seed], outputs=[video_output_0, video_output_1, round1_button, round2_button, multi_button])
    round2_button.click(chatwm.generate_video_next_round2, inputs=[text_input, ddim_steps, fs, 
                                                      n_samples,unconditional_guidance_scale, ddim_eta,random_seed], outputs=[video_output_0, video_output_2,round2_button, round3_button])
    round3_button.click(chatwm.generate_video_next_round3, inputs=[text_input, ddim_steps, fs, 
                                                      n_samples,unconditional_guidance_scale, ddim_eta,random_seed], outputs=[video_output_0, video_output_3,round3_button, round4_button])
    round4_button.click(chatwm.generate_video_next_round4, inputs=[text_input, ddim_steps, fs, 
                                                        n_samples,unconditional_guidance_scale, ddim_eta,random_seed], outputs=[video_output_0, video_output_4,round4_button, round5_button])
    round5_button.click(chatwm.generate_video_next_round5, inputs=[text_input, ddim_steps, fs, 
                                                        n_samples,unconditional_guidance_scale, ddim_eta,random_seed], outputs=[video_output_0, video_output_5, round5_button, round1_button])
    multi_button.click(chatwm.generate_video_mutliround, inputs=[image_input, text_input, ddim_steps, fs, 
                                                      n_samples,unconditional_guidance_scale, ddim_eta, num_round], outputs=[video_output_0,round2_button,round3_button,round4_button,round5_button])
    clear_button.click(gradio_reset,inputs=[random_seed],outputs=total_output)
# demo.queue()
demo.launch(share=True)

#!/usr/bin/env python

import argparse
import datetime
import os
import os.path as osp
import time
import traceback
import warnings

import cv2
import gradio as gr
import numpy as np
from mmengine.config import Config, DictAction
from mmengine.runner.utils import set_random_seed
from PIL import Image
from xtuner.configs import cfgs_name_path
from xtuner.tools.utils import set_model_resource

from xsam.dataset.utils.coco import COCO_INSTANCE_CATEGORIES, COCO_SEMANTIC_CATEGORIES
from xsam.demo.demo import XSamDemo
from xsam.utils.logging import print_log, set_default_logging_format
from xsam.utils.utils import register_function

this_dir = osp.dirname(osp.abspath(__file__))

# Global setup
set_default_logging_format()
warnings.filterwarnings("ignore")

# Custom CSS for better styling
custom_css = """
/* 全局样式 */
.gradio-container {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #1B5E20;
    min-height: 100vh;
}

/* 主容器 */
.main {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    margin: 15px;
    padding: 20px;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
}

/* 主标题样式 */
.main-header {
    text-align: center;
    background: #1B5E20;
    color: white;
    padding: 3rem 2rem;
    border-radius: 20px;
    margin-bottom: 3rem;
    box-shadow: 0 15px 35px rgba(27, 94, 32, 0.3);
    position: relative;
    overflow: hidden;
}

/* 徽章样式优化 */
.main-header div[onclick] {
    display: inline-block !important;
    cursor: pointer !important;
    position: relative !important;
    z-index: 1000 !important;
    transition: transform 0.2s ease;
}

.main-header img {
    margin: 0 3px;
    border-radius: 6px;
    transition: transform 0.2s ease;
    display: block !important;
}

.main-header div[onclick]:hover {
    transform: scale(1.05);
}

.main-header div[onclick]:hover img {
    transform: scale(1.05);
}

.main-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="25" cy="25" r="2" fill="rgba(255,255,255,0.1)"/><circle cx="75" cy="75" r="3" fill="rgba(255,255,255,0.1)"/><circle cx="50" cy="10" r="1" fill="rgba(255,255,255,0.1)"/></svg>');
}

.main-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 1rem;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    position: relative;
    z-index: 1;
}

.running-info {
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #2196f3;
}

.input-section, .output-section {
    display: flex;
    flex-direction: column;
}

.input-section > div, .output-section > div {
    flex-grow: 1;
}

/* 输入区域样式优化 */
.input-section {
    padding-right: 10px;
}

.output-section {
    padding-left: 10px;
}

/* Video instruction spacing tweaks */
.video-instruction {
    margin: 3px 0 3px 0 !important;
    padding: 0 !important;
}

/* Reduce the gap before the main row below the video instruction */
.main-row {
    margin-top: 6px !important;
}

/* Usage instructions styling */
.usage-instructions {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    border-left: 4px solid #28a745;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
}

.usage-instructions h3 {
    color: #2c3e50;
    margin-top: 0;
    margin-bottom: 15px;
    font-weight: 600;
}

.usage-instructions ul {
    margin: 0;
    padding-left: 0;
}

.usage-instructions li {
    margin-bottom: 8px;
    padding: 8px 0;
    border-bottom: 1px solid rgba(0, 0, 0, 0.05);
}

.usage-instructions li:last-child {
    border-bottom: none;
}

.usage-instructions strong {
    color: #495057;
}

.task-description {
    border-left: 4px solid #007bff;
}

.image-upload, .image-upload > div, .image-upload canvas, .image-upload img {
    width: 100% !important;
    height: 500px !important;
    max-width: 100% !important;
    min-height: 400px !important;
    object-fit: contain !important;
    display: block !important;
}
"""

TASK_DESCRIPTION = {
    "imgconv": "Image conversation - Answer questions about the image",
    "genseg": "General segmentation - Segment objects by category names",
    "refseg": "Referring segmentation - Segment objects by referring expressions",
    "reaseg": "Reasoning segmentation - Segment objects by reasoning questions",
    "gcgseg": "GCG segmentation - Generate caption then segment objects in the caption",
    "interseg": "Interactive segmentation - Segment objects by the interactive prompt",
    "vgdseg": "VGD segmentation - Segment objects by the visual grounded prompt",
}

SUPPORTED_TASKS = list(TASK_DESCRIPTION.keys())

# Examples with proper image paths
EXAMPLES = {
    "imgconv": [
        (
            osp.join(this_dir, "./images/imgconv.jpg")
            if osp.exists(osp.join(this_dir, "./images/imgconv.jpg"))
            else None
        ),
        "Can you describe this image briefly? Please elaborate on your response.",
        "imgconv",
    ],
    "genseg": [
        (osp.join(this_dir, "./images/genseg.jpg") if osp.exists(osp.join(this_dir, "./images/genseg.jpg")) else None),
        "ins: "
        + ", ".join([c["name"] for c in COCO_INSTANCE_CATEGORIES])
        + ";\nsem: "
        + ", ".join([c["name"] for c in COCO_SEMANTIC_CATEGORIES]),
        "genseg",
    ],
    "refseg": [
        (osp.join(this_dir, "./images/refseg.jpg") if osp.exists(osp.join(this_dir, "./images/refseg.jpg")) else None),
        "the white tshirt kid",
        "refseg",
    ],
    "reaseg": [
        (osp.join(this_dir, "./images/reaseg.jpg") if osp.exists(osp.join(this_dir, "./images/reaseg.jpg")) else None),
        "What object can be put into dog food?",
        "reaseg",
    ],
    "gcgseg": [
        (osp.join(this_dir, "./images/gcgseg.jpg") if osp.exists(osp.join(this_dir, "./images/gcgseg.jpg")) else None),
        "Can you provide a brief description of this image? Please respond with interleaved segmentation masks for the corresponding phrases.",
        "gcgseg",
    ],
    "interseg": [
        (
            osp.join(this_dir, "./images/interseg.jpg")
            if osp.exists(osp.join(this_dir, "./images/interseg.jpg"))
            else None
        ),
        "You DON'T NEED to input any prompt for interseg. Draw the object you want to segment on the image (support single object for now).",
        "interseg",
    ],
    "vgdseg": [
        (osp.join(this_dir, "./images/vgdseg.jpg") if osp.exists(osp.join(this_dir, "./images/vgdseg.jpg")) else None),
        "You DON'T NEED to input any prompt for vgdseg. Draw the object you want to segment on the image (support single and multiple objects).",
        "vgdseg",
    ],
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="X-SAM Gradio Demo")
    parser.add_argument("config", help="config file name or path")
    parser.add_argument("--work-dir", help="directory to save logs and visualizations")
    parser.add_argument(
        "--pth_model",
        type=str,
        default=None,
        help="path to model checkpoint or 'latest' to use the latest checkpoint in work_dir",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--log-dir", type=str, default="./logs", help="directory to save logs")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config options, format: xxx=yyy",
    )
    parser.add_argument("--port", type=int, default=7860, help="port for gradio server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="host for gradio server")
    parser.add_argument("--share", action="store_true", help="share gradio app")
    return parser.parse_args()


class GradioApp:
    def __init__(self, demo: XSamDemo, log_dir: str):
        self.demo = demo
        self.log_dir = log_dir
        self.processing_status = "Ready"

    def gradio_predict_with_progress(self, data, prompt, task_name="imgconv", score_thr=0.5, progress=gr.Progress()):
        """Enhanced prediction function with progress tracking and better error handling"""
        if data is None:
            return "❌ No image provided", "", "", None

        try:
            progress(0.1, desc="🔍 Initializing...")

            # Validate inputs
            if not prompt or prompt.strip() == "":
                if task_name not in ["gcgseg", "interseg", "vgdseg"]:  # gcgseg doesn't need prompt
                    return "❌ No prompt provided", "", "", None

            # Logging setup
            day_timestamp = datetime.datetime.now().strftime("%Y%m%d")
            file_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            day_log_dir = osp.join(self.log_dir, day_timestamp)
            log_file = osp.join(day_log_dir, f"{day_timestamp}.log")
            img_log_dir = osp.join(day_log_dir, "image")
            out_log_dir = osp.join(day_log_dir, "output")

            os.makedirs(day_log_dir, exist_ok=True)
            os.makedirs(img_log_dir, exist_ok=True)
            os.makedirs(out_log_dir, exist_ok=True)

            progress(0.3, desc="🖼️ Processing image...")

            # Convert PIL image to format expected by demo
            vprompt_masks = None
            if isinstance(data, Image.Image):
                pil_image = data
                array_image = np.array(pil_image)
            elif isinstance(data, np.ndarray):
                pil_image = Image.fromarray(data)
                array_image = data
            elif isinstance(data, dict):
                pil_image = data["background"].convert("RGB")
                array_image = np.array(pil_image)
                vprompt_masks = [np.array(layer)[..., -1] for layer in data["layers"]]
                vprompt_masks = [mask for mask in vprompt_masks if mask.sum() > 0]
                vprompt_masks = None if len(vprompt_masks) == 0 else vprompt_masks
            else:
                raise ValueError(f"Unsupported image type: {type(data)}")

            progress(0.5, desc="🔎 Running X-SAM...")

            # Run prediction using custom logic
            start_time = time.time()

            # Run model inference
            llm_input, llm_output, seg_output = self.demo.run_on_image(
                pil_image, prompt, task_name, vprompt_masks=vprompt_masks, threshold=score_thr
            )

            llm_success = llm_output is not None
            seg_success = seg_output is not None

            inference_time = time.time() - start_time

            progress(0.9, desc="💾 Saving results...")
            # Save input image and output image
            cv2.imwrite(f"{img_log_dir}/{file_timestamp}.png", cv2.cvtColor(array_image, cv2.COLOR_RGB2BGR))
            if seg_success:
                cv2.imwrite(f"{out_log_dir}/{file_timestamp}.png", cv2.cvtColor(seg_output, cv2.COLOR_RGB2BGR))

            # Log to file
            if not osp.exists(log_file):
                with open(log_file, "w") as f:
                    f.write("timestamp\timage\tprompt\ttask_name\tinference_time\tllm_success\tseg_success\n")
            with open(log_file, "a") as f:
                f.write(
                    f"{file_timestamp}\t{file_timestamp}.png\t{prompt}\t{task_name}\t{inference_time:.3f}\t{llm_success}\t{seg_success}\n"
                )

            progress(1.0, desc="✅ Complete!")

            if llm_success or seg_success:
                status_message = f"✅ Completed in {inference_time:.2f}s."
            else:
                status_message = f"⚠️ Failed in {inference_time:.2f}s."

            return (
                status_message,
                llm_input,
                llm_output,
                (gr.update(value=seg_output, height=seg_output.shape[0] + 10) if seg_output is not None else None),
            )

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            print(f"Error in gradio_predict: {traceback.format_exc()}")
            return error_msg, "", "", None

    def create_interface(self):
        # Process examples to load images for ImageEditor format
        examples = []
        for _, example_data in EXAMPLES.items():
            if example_data and len(example_data) >= 3:
                image_path, text_prompt, task_name = example_data

                # Load image if path exists
                if image_path and osp.exists(image_path):
                    try:
                        image = Image.open(image_path).convert("RGBA")
                        mask = Image.fromarray(np.zeros((image.height, image.width, 4), dtype=np.uint8)).convert(
                            "RGBA"
                        )
                        examples.append(
                            [{"background": image, "layers": [mask], "composite": image}, text_prompt, task_name]
                        )
                    except Exception as e:
                        print(f"Error loading example image {image_path}: {e}\n{traceback.format_exc()}")
                        continue
                else:
                    # Skip examples with missing images
                    continue

        with gr.Blocks(title="X-SAM", css=custom_css, theme=gr.themes.Soft()) as app:
            # Header
            gr.HTML(
                """
                <div class="main-header">
                    <h1>✨X-SAM</h1>
                    <h2>From Segment Anything to Any Segmentation</h2>
                    <div style="display: flex; justify-content: center; align-items: center; margin-top: 20px;">
                        <div onclick="window.open('http://arxiv.org/abs/2508.04655', '_blank')" style="margin: 0 2px; cursor: pointer; display: inline-block;">
                            <img src='https://img.shields.io/badge/arXiv-2508.04655-red?style=flat&logo=arXiv&logoColor=red' alt='arxiv' style="display: block;">
                        </div>
                        <div onclick="window.open('https://huggingface.co/hao9610/X-SAM', '_blank')" style="margin: 0 2px; cursor: pointer; display: inline-block;">
                            <img src='https://img.shields.io/badge/HuggingFace-ckpts-orange?style=flat&logo=HuggingFace&logoColor=orange' alt='huggingface' style="display: block;">
                        </div>
                        <div onclick="window.open('https://github.com/wanghao9610/X-SAM', '_blank')" style="margin: 0 2px; cursor: pointer; display: inline-block;">
                            <img src='https://img.shields.io/badge/GitHub-Repo-blue?style=flat&logo=GitHub' alt='GitHub' style="display: block;">
                        </div>
                        <div onclick="window.open('http://47.115.200.157:7861', '_blank')" style="margin: 0 2px; cursor: pointer; display: inline-block;">
                            <img src='https://img.shields.io/badge/Demo-Gradio-gold?style=flat&logo=Gradio&logoColor=red' alt='Demo' style="display: block;">
                        </div>
                        <div onclick="window.open('https://wanghao9610.github.io/X-SAM/', '_blank')" style="margin: 0 2px; cursor: pointer; display: inline-block;">
                            <img src='https://img.shields.io/badge/🌐_Project-Webpage-green?style=flat&logoColor=white' alt='webpage' style="display: block;">
                        </div>
                    </div>
                </div>
            """
            )

            # Main X-SAM Tab
            with gr.Row(elem_classes="main-row"):
                with gr.Column(scale=5, elem_classes="input-section"):
                    # Image input with preview
                    image_input = gr.ImageEditor(
                        type="pil",
                        label="📸 Input Image",
                        elem_classes="image-upload",
                        brush=gr.Brush(
                            colors=[
                                "#FF0000",
                                "#00FF00",
                                "#0000FF",
                                "#FF00FF",
                                "#00FFFF",
                            ],
                            default_color="#FF0000",
                            default_size=5,
                        ),
                        eraser=gr.Eraser(default_size=5),
                        sources=["upload", "webcam", "clipboard"],
                    )

                    # Enhanced text input with suggestions
                    with gr.Group(elem_classes="prompt-group"):
                        # Task selection
                        task_name = gr.Dropdown(
                            choices=SUPPORTED_TASKS,
                            value="imgconv",
                            label="🎯 Task Name",
                            info="Choose the name of task you want to perform",
                            elem_classes="task-dropdown",
                        )

                        # Task description
                        task_description = gr.Textbox(
                            value=TASK_DESCRIPTION["imgconv"],
                            label="📋 Task Description",
                            interactive=False,
                            lines=1,
                            elem_classes="task-description",
                        )

                        # Load example button
                        suggestions_btn = gr.Button(
                            "💡 Load Example (Image + Prompt)", size="sm", elem_classes="btn-secondary example-btn"
                        )
                        score_thr = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0.5,
                            step=0.01,
                            interactive=True,
                            label="🔍 Score Threshold",
                            elem_classes="score-threshold",
                        )

                        # User prompt input
                        text_input = gr.Textbox(
                            lines=1,
                            label="🤔 User Prompt",
                            placeholder="Enter your prompt here based on the task selected below, please refer to the examples below.",
                            value="",
                            elem_id="user-prompt-input",
                            elem_classes="prompt-input",
                        )

                    # Action buttons
                    with gr.Row(elem_classes="action-buttons"):
                        submit_btn = gr.Button(
                            "🚀 Run X-SAM", variant="primary", size="lg", elem_classes="btn-primary run-btn"
                        )
                        clear_btn = gr.Button(
                            "🗑️ Clear All", variant="secondary", elem_classes="btn-secondary clear-btn"
                        )

                with gr.Column(scale=6, elem_classes="output-section"):
                    # Status indicator at top
                    status_display = gr.Textbox(
                        value="🟢 Ready to process - Upload an image and enter a prompt to get started!",
                        label="ℹ️ Running Info",
                        interactive=False,
                        elem_classes="running-info status-display",
                        lines=1,
                    )

                    # LLM Interaction
                    with gr.Group(elem_classes="llm-section"):
                        gr.HTML("<h3 style='margin: 0 0 15px 0; color: #FFFFFF;'>🤖 Conversation</h3>")

                        llm_input = gr.Textbox(
                            value="",
                            label="📝 Language Instruction",
                            placeholder="The language instruction will be displayed here.",
                            lines=1,
                            elem_classes="llm-input",
                            interactive=False,
                        )
                        llm_output = gr.Textbox(
                            value="",
                            label="💬 Language Response",
                            placeholder="The language response will be displayed here.",
                            lines=1,
                            elem_classes="llm-output",
                            interactive=False,
                        )

                    # Segmentation Results
                    with gr.Group(elem_classes="seg-section"):
                        seg_output = gr.Image(
                            type="pil",
                            label="🎨 Segmentation Mask",
                            elem_classes="seg-output",
                            height=615,
                        )

            gr.HTML(
                """
                <div>
                    <span style="font-size: 1.15rem; color: white; font-weight: 600;">
                        🎥 Video Instruction 👉 <a href="https://github.com/user-attachments/assets/1a21cf21-c0bb-42cd-91c8-290324b68618" target="_blank" style="color: white; text-decoration: underline;"> Here </a>
                    </span>
                </div>
                """,
                elem_classes="video-instruction",
            )

            # Examples Section
            if examples:  # Only show if we have valid examples
                with gr.Group(elem_classes="examples-section"):
                    gr.HTML("<h3 style='margin: 0 0 20px 0; text-align: center;'>🌟 Example Gallery</h3>")
                    gr.Examples(
                        examples=examples,
                        inputs=[image_input, text_input, task_name, 0.5],
                        outputs=[status_display, llm_input, llm_output, seg_output],
                        fn=self.gradio_predict_with_progress,
                        cache_examples=False,
                        examples_per_page=10,
                    )

            # Event handlers
            submit_btn.click(
                fn=self.gradio_predict_with_progress,
                inputs=[image_input, text_input, task_name, score_thr],
                outputs=[status_display, llm_input, llm_output, seg_output],
                show_progress=True,
            )

            clear_btn.click(
                fn=lambda: [
                    None,
                    "",
                    "imgconv",
                    "",
                    None,
                    gr.update(value=None, height=615),
                    0.5,
                    "🧹 All cleared! Ready for new input - Upload an image and enter a prompt to get started!",
                ],
                outputs=[
                    image_input,
                    text_input,
                    task_name,
                    llm_input,
                    llm_output,
                    seg_output,
                    score_thr,
                    status_display,
                ],
            )

            suggestions_btn.click(fn=self.get_examples, inputs=[task_name], outputs=[image_input, text_input])

            task_name.change(
                fn=lambda task: TASK_DESCRIPTION.get(task, ""),
                inputs=[task_name],
                outputs=[task_description],
            )

            # Auto-update status when image is uploaded
            image_input.change(
                fn=lambda data: (
                    "📸 Image uploaded successfully! Enter a prompt and click 'Run X-SAM' to begin analysis."
                    if data is not None
                    else "🟢 Ready to process - Upload an image and enter a prompt to get started!"
                ),
                inputs=[image_input],
                outputs=[status_display],
            )

        return app

    def get_examples(self, task_name):
        """Get examples for the given task - returns image and text prompt"""
        example = EXAMPLES.get(task_name, None)
        if not example:
            return {"background": None, "layers": [None], "composite": None}, ""
        try:
            image_path = example[0]
            text_prompt = example[1]

            # Load image if path exists
            if image_path and osp.exists(image_path):
                try:
                    image = Image.open(image_path).convert("RGBA")
                    mask = Image.fromarray(np.zeros((image.height, image.width, 4), dtype=np.uint8)).convert("RGBA")
                    return {"background": image, "layers": [mask], "composite": image}, text_prompt
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}\n{traceback.format_exc()}")
                    return {"background": None, "layers": [None], "composite": None}, text_prompt
            else:
                return {"background": None, "layers": [None], "composite": None}, text_prompt

        except Exception as e:
            print(f"Error processing example for task {task_name}: {e}\n{traceback.format_exc()}")
            return {"background": None, "layers": [None], "composite": None}, ""


def setup_cfg(args):
    """Setup configuration from arguments."""
    # Load and process config
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f"Cannot find {args.config}")

    cfg = Config.fromfile(args.config)
    set_model_resource(cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.seed is not None:
        set_random_seed(args.seed)
        print_log(f"Set the random seed to {args.seed}.", logger="current")
    register_function(cfg._cfg_dict)

    # Handle latest checkpoint
    if args.pth_model == "latest":
        from mmengine.runner import find_latest_checkpoint

        if args.work_dir and osp.exists(osp.join(args.work_dir, "pytorch_model.bin")):
            args.pth_model = osp.join(args.work_dir, "pytorch_model.bin")
        elif args.work_dir:
            args.pth_model = find_latest_checkpoint(args.work_dir)
        else:
            raise ValueError("work_dir must be specified when using 'latest' checkpoint")
        print_log(f"Found latest checkpoint: {args.pth_model}", logger="current")

    return args, cfg


def main():
    """Main function for X-SAM Gradio demo."""
    args = parse_args()

    # Setup configuration
    args, cfg = setup_cfg(args)

    # Create demo instance
    print_log("Initializing X-SAM demo...", logger="current")
    demo = XSamDemo(cfg, args.pth_model, output_ids_with_output=False)
    print_log("X-SAM demo initialized successfully!", logger="current")

    # Create Gradio app
    gradio_app = GradioApp(demo, args.log_dir)
    app = gradio_app.create_interface()

    # Launch the app
    print_log(f"Starting Gradio server on {args.host}:{args.port}", logger="current")
    app.launch(
        show_error=True,
        share=args.share,
        server_port=args.port,
        server_name=args.host,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python

import argparse
import datetime
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import os.path as osp
import time
import traceback
import warnings

import gradio as gr
import numpy as np
from mmengine.config import Config, DictAction
from mmengine.runner.utils import set_random_seed
from PIL import Image

from xsam.dataset.utils.coco import COCO_INSTANCE_CATEGORIES, COCO_SEMANTIC_CATEGORIES
from xsam.dataset.utils.load import load_image
from xsam.demo.demo import XSamDemo
from xsam.utils.configs import cfgs_name_path
from xsam.utils.logging import print_log, set_default_logging_format
from xsam.utils.utils import register_function, set_model_resource

this_dir = osp.dirname(osp.abspath(__file__))

# Global setup
set_default_logging_format()
warnings.filterwarnings("ignore")

# Custom CSS inspired by the project webpage
custom_css = """
/* ── Force light mode globally – prevents Gradio dark theme ── */
html, body {
    color-scheme: light !important;
    background: #f0f4f8 !important;
}

:root {
    color-scheme: light !important;
    --primary: #10b981;
    --primary-dark: #059669;
    --primary-light: #d1fae5;
    --accent: #0ea5e9;
    --bg-body: #f0f4f8;
    --bg-soft: #f0fdf4;
    --bg-card: #ffffff;
    --bg-panel: #f8fafc;
    --text-main: #0f172a;
    --text-muted: #475569;
    --border-color: #e2e8f0;
    --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
    --shadow-md: 0 10px 15px -3px rgb(0 0 0 / 0.08), 0 4px 6px -4px rgb(0 0 0 / 0.08);
    --shadow-lg: 0 20px 35px -12px rgba(15, 23, 42, 0.14);
    --shadow-glow: 0 0 22px rgba(16, 185, 129, 0.14);
    --radius-lg: 28px;
    --radius-md: 18px;
    --radius-sm: 12px;
    --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    /* Override Gradio internal CSS variables (light theme) */
    --block-background-fill: #ffffff;
    --input-background-fill: #ffffff;
    --panel-background-fill: #f8fafc;
    --background-fill-primary: #ffffff;
    --background-fill-secondary: #f8fafc;
    --color-background-primary: #ffffff;
    --body-background-fill: #f0f4f8;
    --block-border-color: #e2e8f0;
    --border-color-primary: #e2e8f0;
    --color-border-primary: #e2e8f0;
    --input-border-color: #e2e8f0;
    --block-label-background-fill: transparent;
    --block-label-text-color: #0f172a;
    --input-placeholder-color: #94a3b8;
    --neutral-950: #0f172a;
    --neutral-900: #1e293b;
    --neutral-800: #334155;
    /* Additional Gradio 4.x theme tokens */
    --color-accent: #10b981;
    --button-primary-background-fill: #10b981;
    --button-primary-text-color: #ffffff;
    --button-secondary-background-fill: #ffffff;
    --button-secondary-text-color: #475569;
    --table-even-background-fill: #f8fafc;
    --table-odd-background-fill: #ffffff;
    --table-row-focus: rgba(16, 185, 129, 0.07);
    --checkbox-background-color: #ffffff;
    --slider-color: #10b981;
    --stat-background-fill: #f8fafc;
}

/* Prevent Gradio dark-media-query overrides by re-asserting in prefers-color-scheme */
@media (prefers-color-scheme: dark) {
    :root {
        color-scheme: light !important;
        --block-background-fill: #ffffff !important;
        --input-background-fill: #ffffff !important;
        --panel-background-fill: #f8fafc !important;
        --background-fill-primary: #ffffff !important;
        --background-fill-secondary: #f8fafc !important;
        --color-background-primary: #ffffff !important;
        --body-background-fill: #f0f4f8 !important;
        --block-border-color: #e2e8f0 !important;
        --border-color-primary: #e2e8f0 !important;
        --input-border-color: #e2e8f0 !important;
        --block-label-background-fill: transparent !important;
        --block-label-text-color: #0f172a !important;
        --neutral-950: #0f172a !important;
        --neutral-900: #1e293b !important;
        --neutral-800: #334155 !important;
    }
    html, body {
        background: #f0f4f8 !important;
        color: #0f172a !important;
    }
}

/* ── Global ──────────────────────────────────────────────── */
.gradio-container {
    font-family: 'Plus Jakarta Sans', 'Segoe UI', system-ui, sans-serif !important;
    color: var(--text-main) !important;
    background:
        radial-gradient(circle at top left, rgba(16, 185, 129, 0.07), transparent 32%),
        radial-gradient(circle at top right, rgba(14, 165, 233, 0.07), transparent 30%),
        var(--bg-body) !important;
    min-height: 100vh;
    max-width: 100% !important;
    width: 100% !important;
    padding: 0 !important;
}

.gradio-container .contain {
    max-width: 100% !important;
    width: 100% !important;
    padding: 0 1rem !important;
    margin: 0 auto !important;
    box-sizing: border-box !important;
}

.gradio-container .prose,
.gradio-container .prose * {
    font-family: 'Plus Jakarta Sans', 'Segoe UI', system-ui, sans-serif !important;
    color: var(--text-main) !important;
}

/* ── 1. Remove gray divider line below header ─────────────── */
.gradio-container hr,
.gradio-container .divider {
    display: none !important;
    border: none !important;
    height: 0 !important;
    margin: 0 !important;
}

/* ── Header ──────────────────────────────────────────────── */
.main-header {
    position: relative;
    overflow: hidden;
    margin: 0;
    padding: 5rem 1.5rem 4rem;
    background: linear-gradient(135deg, #ecfdf5 0%, #e0f2fe 100%);
    text-align: center;
    border-bottom: none;
}

.main-header::before {
    content: "";
    position: absolute;
    inset: -50%;
    background: radial-gradient(circle at 50% 30%, rgba(255, 255, 255, 0.85) 0%, rgba(255, 255, 255, 0) 52%);
    animation: pulseGlow 15s ease-in-out infinite alternate;
    pointer-events: none;
}

.main-header-inner {
    position: relative;
    z-index: 1;
    max-width: 980px;
    margin: 0 auto;
}

.main-header h1,
.main-header .publication-title {
    margin: 0 0 0.75rem;
    font-size: clamp(2.8rem, 5vw, 4.2rem);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.04em;
    color: var(--primary-dark) !important;
    animation: fadeInUp 0.7s ease-out both;
}

/* ── 4. Subtitle visibility fix ──────────────────────────── */
.main-header h2,
.main-header .publication-subtitle {
    margin: 0 0 2rem;
    font-size: clamp(1.2rem, 3vw, 1.8rem);
    font-weight: 600;
    line-height: 1.35;
    letter-spacing: -0.01em;
    color: var(--primary-dark) !important;
    animation: fadeInUp 0.7s ease-out 0.08s both;
}

@supports ((-webkit-background-clip: text) or (background-clip: text)) {
    .main-header .publication-title,
    .main-header .publication-subtitle {
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary) 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }
}

@media (prefers-color-scheme: dark) {
    .main-header .publication-title,
    .main-header .publication-subtitle {
        background: none !important;
        color: #10b981 !important;
        -webkit-text-fill-color: #10b981 !important;
    }
}

.dark .main-header .publication-title,
.dark .main-header .publication-subtitle,
[data-theme="dark"] .main-header .publication-title,
[data-theme="dark"] .main-header .publication-subtitle {
    background: none !important;
    color: #10b981 !important;
    -webkit-text-fill-color: #10b981 !important;
}

.hero-authors,
.hero-affiliations {
    animation: fadeInUp 0.7s ease-out 0.16s both;
}

.hero-authors {
    margin-bottom: 0.9rem;
    font-size: 1.04rem;
    color: #334155;
}

.hero-authors a {
    display: inline-block;
    margin: 0.3rem 0.6rem;
    color: #1e293b;
    text-decoration: none;
    font-weight: 700;
    transition: var(--transition);
}

.hero-authors a:hover {
    color: var(--primary-dark);
    transform: translateY(-1px);
}

.hero-affiliations {
    margin-bottom: 1.8rem;
    font-size: 0.96rem;
    font-weight: 500;
    color: #334155;
}

.hero-affiliations span {
    display: inline-block;
    margin: 0.15rem 0.7rem;
}

.hero-corresponding {
    margin-top: 0.4rem;
    font-size: 0.86rem;
    color: #475569;
    opacity: 0.9;
}

.header-links {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.9rem;
    animation: fadeInUp 0.7s ease-out 0.24s both;
}

.header-links a {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem 1.3rem;
    border-radius: 999px;
    border: 1px solid rgba(16, 185, 129, 0.3);
    background: rgba(255, 255, 255, 0.88);
    color: #1e293b;
    font-weight: 700;
    text-decoration: none;
    box-shadow: var(--shadow-sm);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    transition: var(--transition);
}

.header-links .link-icon {
    width: 16px;
    height: 16px;
    object-fit: contain;
    flex-shrink: 0;
}

.header-links .link-emoji {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 16px;
    height: 16px;
    font-size: 0.95rem;
    line-height: 1;
    flex-shrink: 0;
}

.header-links a:hover {
    background: var(--primary);
    color: #ffffff;
    border-color: var(--primary);
    transform: translateY(-2px);
    box-shadow: 0 10px 20px rgba(16, 185, 129, 0.2);
}

/* ── Layout ──────────────────────────────────────────────── */
.main-row {
    gap: 1.5rem;
    align-items: stretch !important;
    margin-top: 1.75rem;
    width: 100% !important;
}

.main-row > * {
    min-width: 0 !important;
    align-self: stretch !important;
}

.input-section,
.output-section,
.examples-section {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-lg) !important;
    box-shadow: var(--shadow-md) !important;
    transition: var(--transition);
}

.input-section,
.output-section {
    padding: 1.25rem !important;
    width: 100% !important;
    box-sizing: border-box !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 0.85rem;
}

.output-section .mask-panel {
    flex: 0 0 auto !important;
    display: flex !important;
    flex-direction: column !important;
    min-height: 0 !important;
}

.output-section .mask-panel > *,
.output-section .mask-panel .block,
.output-section .mask-panel .form,
.output-section .mask-panel .wrap {
    min-height: 0 !important;
}

.output-section .mask-panel .mask-media,
.output-section .mask-panel .mask-media > .block,
.output-section .mask-panel .mask-media [data-testid="image"],
.output-section .mask-panel .mask-media .image-container,
.output-section .mask-panel .mask-media .wrap {
    flex: 1 1 auto !important;
    height: 100% !important;
    width: 100% !important;
    max-height: none !important;
    min-height: 320px !important;
}

.output-section .mask-panel .mask-media .image-container,
.output-section .mask-panel .mask-media .wrap {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

.output-section .mask-panel .mask-media img {
    width: 100% !important;
    height: 100% !important;
    object-fit: contain !important;
}

.input-section .action-buttons {
    margin-top: auto !important;
}

.input-section:hover,
.output-section:hover,
.examples-section:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg), var(--shadow-glow) !important;
}

/* ── 2 & 3. Panel / card backgrounds (light, not dark) ───── */
.panel-section {
    padding: 1rem !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-md) !important;
    background: var(--bg-panel) !important;
    box-shadow: var(--shadow-sm) !important;
    width: 100% !important;
    box-sizing: border-box !important;
}

.media-card {
    padding: 0.85rem !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-md) !important;
    background: #ffffff !important;
    box-shadow: var(--shadow-sm) !important;
    width: 100% !important;
    box-sizing: border-box !important;
}

.controls-card {
    padding: 1rem !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-md) !important;
    background: #ffffff !important;
    box-shadow: var(--shadow-sm) !important;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    width: 100% !important;
    box-sizing: border-box !important;
}

/* Force all direct children and descendants of controls-card to light backgrounds */
body .controls-card,
body .controls-card > *,
body .controls-card > * > *,
body .controls-card .block,
body .controls-card .form,
body .controls-card .wrap,
body .controls-card .gap,
body .controls-card [class*="svelte-"] {
    background: #ffffff !important;
    color: #0f172a !important;
}

.task-row {
    gap: 0.75rem;
    align-items: end;
    background: transparent !important;
}
.task-row .gap,
.task-row .wrap,
.task-row .form,
.task-row .block,
.task-row [class*="svelte-"] {
    background: transparent !important;
    box-shadow: none !important;
    border-color: transparent !important;
    color: #0f172a !important;
}

.section-title {
    margin: 0 0 0.75rem !important;
    font-size: 1.02rem !important;
    font-weight: 800 !important;
    color: var(--text-main) !important;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.section-title .icon {
    color: var(--primary);
}

.gradio-container .section-title::after,
.gradio-container h3:has(.section-title)::after {
    content: none !important;
    display: none !important;
    border: none !important;
    background: none !important;
}

.prompt-group {
    padding: 0 !important;
    border: none !important;
    border-radius: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

/* ── Dropdown options popup ──────────────────────────────── */
.gradio-container [data-testid="dropdown"] > div:last-child,
.gradio-container [data-testid="dropdown"] ul,
.gradio-container [data-testid="dropdown"] .options,
.gradio-container [data-testid="dropdown"] [role="listbox"],
.gradio-container [data-testid="dropdown"] [role="option"] {
    background: #ffffff !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border-color) !important;
}

.gradio-container [data-testid="dropdown"] [role="option"]:hover,
.gradio-container [data-testid="dropdown"] [role="option"][aria-selected="true"] {
    background: #f0fdf4 !important;
    color: var(--primary-dark) !important;
}

/* ── Slider track & thumb ────────────────────────────────── */
.gradio-container input[type="range"] {
    accent-color: var(--primary) !important;
    background: transparent !important;
}

.gradio-container .gradio-slider .wrap,
.gradio-container [data-testid="slider"] .wrap,
.gradio-container [data-testid="slider"] .block {
    background: transparent !important;
}

/* ── Status box ──────────────────────────────────────────── */
.running-info textarea {
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%) !important;
    border: 1px solid #a7f3d0 !important;
    color: #065f46 !important;
    border-radius: 16px !important;
    font-weight: 700 !important;
    text-align: center !important;
    box-shadow: 0 4px 16px rgba(16, 185, 129, 0.1) !important;
}

/* ── Image upload areas ──────────────────────────────────── */
.image-upload {
    overflow: hidden;
    border: 1.5px dashed var(--border-color) !important;
    border-radius: var(--radius-md) !important;
    background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%) !important;
    transition: var(--transition) !important;
    box-shadow: none !important;
}

.image-upload:hover {
    border-color: rgba(16, 185, 129, 0.5) !important;
    border-style: solid !important;
    box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.08) !important;
}

.image-upload > div,
.image-upload canvas,
.image-upload img {
    border-radius: calc(var(--radius-md) - 2px) !important;
}

/* ── Override ImageEditor dark canvas background to match light theme ── */
.image-upload .image-editor,
.image-upload .image-editor > div,
.image-upload .canvas-wrap,
.image-upload .canvas-container,
.image-upload .tool-wrap,
.image-upload [data-testid="image-editor"],
.image-upload [data-testid="image-editor"] > div {
    background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%) !important;
}

/* Allow ImageEditor brush toolbar to be visible (not clipped by overflow:hidden) */
.image-upload [data-testid="image-editor"],
.image-upload .image-editor {
    overflow: visible !important;
}

.image-upload .image-editor canvas {
    background: transparent !important;
}

/* ── 2. Override block backgrounds (prevent dark bleed-through) ── */
.gradio-container .block,
.gradio-container .gr-panel,
.gradio-container .gr-group,
.gradio-container .gr-box,
.gradio-container .form,
.gradio-container .wrap,
.gradio-container .panel,
body .gradio-container .block,
body .gradio-container .form,
body .gradio-container .wrap {
    background: #ffffff !important;
    border-color: transparent !important;
    box-shadow: none !important;
    color: #0f172a !important;
}

/* Specific containers that should stay transparent */
.input-section .form,
.output-section .form,
.controls-card .form,
.panel-section .form,
.media-card .form,
.prompt-group .form {
    background: transparent !important;
}

/* Advanced settings inner area keeps the green-tint */
.advanced-settings-group .form {
    background: #f0fdf4 !important;
}

/* Image placeholder areas - light background */
.gradio-container [data-testid="image"],
.gradio-container [data-testid="image"] .wrap {
    background: #f1f5f9 !important;
    border-radius: var(--radius-sm) !important;
}

.output-section .mask-panel,
.output-section .mask-panel .block,
.output-section .mask-panel .wrap,
.output-section .mask-panel [data-testid="image"] {
    background: var(--bg-panel) !important;
    border-color: transparent !important;
    box-shadow: none !important;
    overflow: hidden !important;
}

.output-section .mask-panel,
.output-section .mask-panel * {
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}

.output-section .mask-panel::-webkit-scrollbar,
.output-section .mask-panel *::-webkit-scrollbar {
    width: 0 !important;
    height: 0 !important;
    display: none !important;
}

/* ── 5. Task / Description label & content styling ────────── */
.gradio-container .block_label,
.gradio-container .block-title {
    font-weight: 700 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 0 4px !important;
}

.gradio-container .block_label span,
.gradio-container .block-title span,
.gradio-container label span {
    color: var(--text-main) !important;
    background: transparent !important;
    font-weight: 700 !important;
}

.task-select .block_label span,
.task-select label span {
    color: var(--primary-dark) !important;
    background: transparent !important;
    font-weight: 700 !important;
}

.description-box .block_label span,
.description-box label span {
    color: var(--primary-dark) !important;
    background: transparent !important;
    font-weight: 700 !important;
}

.controls-card .task-row .task-select,
.controls-card .task-row > .task-select,
.gradio-container .controls-card .task-row .task-select,
.gradio-container .task-row > .task-select {
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    box-sizing: border-box !important;
}

/* ── Apply the green dashed look to the OUTER dropdown bar only ── */
.controls-card .task-row .task-select .wrap,
.gradio-container .controls-card .task-row .task-select .wrap {
    border: 1.5px dashed #a7f3d0 !important;
    border-radius: var(--radius-md) !important;
    background: #f0fdf4 !important;
    color: var(--primary-dark) !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 3px 0 rgba(16, 185, 129, 0.08) !important;
    transition: all 0.15s ease !important;
    box-sizing: border-box !important;
}

.controls-card .task-row .task-select .wrap > *:not(ul):not(.options),
.controls-card .task-row .task-select .wrap-inner,
.controls-card .task-row .task-select .secondary-wrap,
.controls-card .task-row .task-select input,
.controls-card .task-row .task-select [data-testid="dropdown"]:not(ul),
.gradio-container .controls-card .task-row .task-select .wrap > *:not(ul):not(.options),
.gradio-container .controls-card .task-row .task-select .wrap-inner,
.gradio-container .controls-card .task-row .task-select .secondary-wrap,
.gradio-container .controls-card .task-row .task-select input,
.gradio-container .controls-card .task-row .task-select [data-testid="dropdown"]:not(ul) {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    outline: none !important;
}

/* Hover / focus lit state on the outer bar */
.controls-card .task-row .task-select .wrap:hover,
.controls-card .task-row .task-select .wrap:focus-within,
.gradio-container .controls-card .task-row .task-select .wrap:hover,
.gradio-container .controls-card .task-row .task-select .wrap:focus-within {
    border-style: solid !important;
    border-color: var(--primary) !important;
    background: #ecfdf5 !important;
    box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.15), 0 2px 6px rgba(16, 185, 129, 0.12) !important;
}

.controls-card .task-row > .description-box,
.gradio-container .task-row > .description-box {
    padding: 0.85rem 0.9rem !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: var(--radius-md) !important;
    background: #ffffff !important;
    box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.08), 0 1px 2px 0 rgba(15, 23, 42, 0.04) !important;
    box-sizing: border-box !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}

.controls-card .task-row > .description-box:hover,
.gradio-container .task-row > .description-box:hover {
    border-color: #94a3b8 !important;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08) !important;
}

.controls-card .example-btn,
.gradio-container .controls-card .example-btn {
    border: 1.5px dashed #a7f3d0 !important;
    border-radius: var(--radius-md) !important;
    background: #f0fdf4 !important;
    color: var(--primary-dark) !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 3px 0 rgba(16, 185, 129, 0.08) !important;
    transition: all 0.15s ease !important;
}

.controls-card .example-btn:hover,
.gradio-container .controls-card .example-btn:hover {
    border-style: solid !important;
    border-color: var(--primary) !important;
    background: #ecfdf5 !important;
    box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.15), 0 2px 6px rgba(16, 185, 129, 0.12) !important;
}

.controls-card .prompt-group textarea,
.controls-card .prompt-group input,
.gradio-container .controls-card .prompt-group textarea,
.gradio-container .controls-card .prompt-group input {
    border: 1.5px dashed #a7f3d0 !important;
    border-radius: var(--radius-md) !important;
    background: #f0fdf4 !important;
    color: var(--primary-dark) !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 3px 0 rgba(16, 185, 129, 0.08) !important;
    transition: all 0.15s ease !important;
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}

.controls-card .prompt-group textarea::-webkit-scrollbar,
.controls-card .prompt-group input::-webkit-scrollbar,
.gradio-container .controls-card .prompt-group textarea::-webkit-scrollbar,
.gradio-container .controls-card .prompt-group input::-webkit-scrollbar {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
}

.controls-card .prompt-group textarea::placeholder,
.controls-card .prompt-group input::placeholder,
.gradio-container .controls-card .prompt-group textarea::placeholder,
.gradio-container .controls-card .prompt-group input::placeholder {
    color: rgba(6, 95, 70, 0.55) !important;
    opacity: 1 !important;
}

.controls-card .prompt-group textarea:hover,
.controls-card .prompt-group input:hover,
.gradio-container .controls-card .prompt-group textarea:hover,
.gradio-container .controls-card .prompt-group input:hover {
    border-style: solid !important;
    border-color: var(--primary) !important;
    background: #ecfdf5 !important;
    box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.15), 0 2px 6px rgba(16, 185, 129, 0.12) !important;
}

.controls-card .prompt-group textarea:focus,
.controls-card .prompt-group input:focus,
.gradio-container .controls-card .prompt-group textarea:focus,
.gradio-container .controls-card .prompt-group input:focus {
    border-style: solid !important;
    border-color: var(--primary) !important;
    background: #ecfdf5 !important;
    box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.15), 0 2px 6px rgba(16, 185, 129, 0.12) !important;
}

.task-select .block,
.task-select .form,
.task-select .wrap,
.task-select [class*="svelte-"],
.description-box .block,
.description-box .form,
.description-box .wrap,
.description-box [class*="svelte-"] {
    background: transparent !important;
    box-shadow: none !important;
    border-color: transparent !important;
    color: #0f172a !important;
}

/* Allow dropdown list to escape all parent clipping boundaries */
.controls-card,
.task-row,
.task-row > *,
.task-select,
.task-select .block,
.task-select .form,
.task-select .wrap,
.task-select label {
    overflow: visible !important;
}

/* Anchor the label as the positioning parent for the options list */
.task-select label {
    position: relative !important;
    display: block !important;
}

/* Force task dropdown list to appear directly below the trigger.
   High specificity prefixes are required so this beats the `.wrap *` reset rule
   that sits earlier in the stylesheet. */
.gradio-container .controls-card .task-row .task-select ul.options,
.gradio-container .controls-card .task-row .task-select .wrap ul.options,
.controls-card .task-row .task-select ul.options,
.controls-card .task-row .task-select .wrap ul.options {
    position: absolute !important;
    top: 100% !important;
    bottom: auto !important;
    left: 0 !important;
    z-index: 9999 !important;
    max-height: 260px !important;
    overflow-y: auto !important;
    background: #ffffff !important;
    border: 1px solid #a7f3d0 !important;
    border-radius: 10px !important;
    box-shadow: 0 8px 24px rgba(16, 185, 129, 0.12) !important;
    padding: 4px !important;
    margin-top: 4px !important;
}

.gradio-container .controls-card .task-row .task-select ul.options li,
.gradio-container .controls-card .task-row .task-select .wrap ul.options li,
.controls-card .task-row .task-select ul.options li,
.controls-card .task-row .task-select .wrap ul.options li {
    background: #ffffff !important;
    color: #0f172a !important;
    padding: 6px 10px !important;
    border-radius: 6px !important;
}

.gradio-container .controls-card .task-row .task-select ul.options li:hover,
.gradio-container .controls-card .task-row .task-select ul.options li.selected,
.gradio-container .controls-card .task-row .task-select .wrap ul.options li:hover,
.gradio-container .controls-card .task-row .task-select .wrap ul.options li.selected,
.controls-card .task-row .task-select ul.options li:hover,
.controls-card .task-row .task-select ul.options li.selected,
.controls-card .task-row .task-select .wrap ul.options li:hover,
.controls-card .task-row .task-select .wrap ul.options li.selected {
    background: #ecfdf5 !important;
    color: #059669 !important;
}

/* High-specificity rules to ensure task-select inner inputs are readable */
.controls-card .task-row .task-select button,
.controls-card .task-row .task-select input,
.controls-card .task-row .task-select select {
    color: var(--primary-dark) !important;
    font-weight: 600 !important;
    background: transparent !important;
}

.controls-card .task-row .task-select button svg,
.controls-card .task-row .task-select .icon,
.controls-card .task-row .task-select .icon svg {
    color: var(--primary-dark) !important;
    stroke: var(--primary-dark) !important;
    fill: var(--primary-dark) !important;
}

/* High-specificity rules to ensure description text is always readable */
.controls-card .task-row .description-box textarea,
.controls-card .task-row .description-box input {
    color: var(--text-main) !important;
    font-weight: 500 !important;
    background: transparent !important;
}

/* ── Labels & inputs ─────────────────────────────────────── */
.gradio-container label,
.gradio-container .block-title,
.gradio-container .block_label {
    color: var(--text-main) !important;
    font-weight: 700 !important;
}

.gradio-container textarea,
.gradio-container input,
.gradio-container select,
.gradio-container input[type="text"] {
    border-radius: 12px !important;
    border-color: var(--border-color) !important;
    background: #ffffff !important;
    color: var(--text-main) !important;
}

.gradio-container textarea::placeholder,
.gradio-container input::placeholder {
    color: #94a3b8 !important;
    opacity: 1 !important;
}

.gradio-container input[readonly],
.gradio-container textarea[readonly] {
    color: #334155 !important;
    background: #f8fafc !important;
}

.gradio-container textarea:focus,
.gradio-container input:focus,
.gradio-container select:focus {
    border-color: rgba(16, 185, 129, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.08) !important;
}

/* ── Accordion ───────────────────────────────────────────── */
.gradio-container .accordion {
    border-radius: 14px !important;
    border: 1px solid var(--border-color) !important;
    overflow: hidden;
    background: #ffffff !important;
}

.gradio-container .accordion-header {
    background: #f8fafc !important;
    color: var(--text-main) !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}
.gradio-container .accordion-header span,
.gradio-container .accordion-header button,
.gradio-container .accordion-header p {
    color: var(--text-main) !important;
}

/* ── Dropdowns & textboxes ───────────────────────────────── */
.gradio-container [data-testid="dropdown"] button,
.gradio-container [data-testid="textbox"] textarea,
.gradio-container [data-testid="textbox"] input {
    background: #ffffff !important;
    color: var(--text-main) !important;
}

.gradio-container [data-testid="dropdown"] *,
.gradio-container [data-testid="textbox"] * {
    color: var(--text-main) !important;
}

/* ── Auto-resize textboxes to fit content ── */
.auto-resize-textbox textarea,
.gradio-container .auto-resize-textbox textarea,
.gradio-container .controls-card .prompt-group .auto-resize-textbox textarea {
    field-sizing: content !important;
    min-height: 40px !important;
    max-height: 320px !important;
    overflow-y: auto !important;
    resize: none !important;
    height: auto !important;
}

/* ── Buttons ─────────────────────────────────────────────── */
.run-btn {
    background: linear-gradient(135deg, var(--primary) 0%, #34d399 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 800 !important;
    border-radius: 999px !important;
    box-shadow: 0 10px 24px rgba(16, 185, 129, 0.24) !important;
    transition: var(--transition) !important;
}

.run-btn:hover {
    background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary) 100%) !important;
    transform: translateY(-2px);
    box-shadow: 0 16px 28px rgba(16, 185, 129, 0.3) !important;
}

.run-btn:active {
    transform: translateY(0);
}

button:not(.run-btn)[variant="secondary"] {
    border-radius: 999px !important;
    border: 1px solid var(--border-color) !important;
    background: #ffffff !important;
    color: var(--text-muted) !important;
    font-weight: 700 !important;
    transition: var(--transition) !important;
}

button:not(.run-btn)[variant="secondary"]:hover {
    color: var(--primary-dark) !important;
    border-color: rgba(16, 185, 129, 0.4) !important;
    box-shadow: var(--shadow-sm) !important;
}

.action-buttons {
    margin-top: 0.5rem;
    gap: 0.75rem;
}

/* ── Examples ────────────────────────────────────── */
.examples-section {
    margin-top: 1.5rem;
    padding: 1.5rem !important;
    background: var(--bg-card) !important;
}

/* Table wrapper */
.examples-section .table-wrap {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border-color) !important;
    overflow: hidden !important;
    background: #ffffff !important;
}

/* Table itself */
.examples-section table,
.gradio-container .examples table {
    background: #ffffff !important;
    color: var(--text-main) !important;
    border-collapse: collapse !important;
    width: 100% !important;
}

/* Table header */
.examples-section thead,
.examples-section thead tr,
.examples-section th,
.gradio-container .examples thead,
.gradio-container .examples thead tr,
.gradio-container .examples th {
    background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%) !important;
    color: var(--primary-dark) !important;
    font-weight: 700 !important;
    border-bottom: 2px solid #a7f3d0 !important;
}

/* Table header cells */
.examples-section thead th span,
.examples-section thead th,
.gradio-container .examples thead th span,
.gradio-container .examples thead th {
    color: var(--primary-dark) !important;
    background: transparent !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.02em !important;
}

/* Table body rows */
.examples-section tbody,
.examples-section tbody tr,
.gradio-container .examples tbody,
.gradio-container .examples tbody tr {
    background: #ffffff !important;
    color: var(--text-main) !important;
}

/* Alternating rows */
.examples-section tbody tr:nth-child(even),
.gradio-container .examples tbody tr:nth-child(even) {
    background: #f8fafc !important;
}

/* Row hover */
.examples-section tbody tr:hover,
.gradio-container .examples tbody tr:hover {
    background: rgba(16, 185, 129, 0.07) !important;
    cursor: pointer;
}

/* Table cells */
.examples-section td,
.gradio-container .examples td {
    color: var(--text-main) !important;
    background: transparent !important;
    border-bottom: 1px solid var(--border-color) !important;
    padding: 0.6rem 0.75rem !important;
    font-size: 0.9rem !important;
}

/* Cell text */
.examples-section td span,
.examples-section td p,
.gradio-container .examples td span,
.gradio-container .examples td p {
    color: var(--text-main) !important;
    background: transparent !important;
}

/* Pagination & buttons inside examples */
.examples-section .paginate,
.examples-section .paginate button,
.gradio-container .examples .paginate,
.gradio-container .examples .paginate button {
    background: #ffffff !important;
    color: var(--text-main) !important;
    border-color: var(--border-color) !important;
}

.examples-section .paginate button:hover,
.gradio-container .examples .paginate button:hover {
    background: var(--primary-light) !important;
    color: var(--primary-dark) !important;
    border-color: var(--primary) !important;
}

/* Hide "Visual Prompt" (col 2) and "Score Threshold" (col 5) */
.examples-section table th:nth-child(2),
.examples-section table td:nth-child(2),
.examples-section table th:nth-child(5),
.examples-section table td:nth-child(5),
.gradio-container .examples table th:nth-child(2),
.gradio-container .examples table td:nth-child(2),
.gradio-container .examples table th:nth-child(5),
.gradio-container .examples table td:nth-child(5) {
    display: none !important;
}

/* ── Animations ──────────────────────────────────────────── */
@keyframes pulseGlow {
    0% { transform: scale(1) translate(0, 0); }
    100% { transform: scale(1.05) translate(2%, 2%); }
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Responsive ──────────────────────────────────────────── */
@media (max-width: 900px) {
    .main-header {
        padding: 4rem 1rem 3rem;
        margin-left: -12px;
        margin-right: -12px;
    }
    .main-row { gap: 1rem; }
    .task-row { gap: 0.6rem; }
}

/* ── Progress bar description text color fix ─────────────── */
.progress-text,
.progress-text span,
.progress-level,
.progress-level span,
.progress-level-inner,
.generating,
[class*="progress"] span,
[class*="progress"] p,
[class*="progress"] div {
    color: #0f172a !important;
}
"""

TASK_DESCRIPTION = {
    "img_chat": "Image Chat - Answer questions about the image",
    "img_genseg": "Image general segmentation - Segment objects by category names",
    "img_refseg": "Image referring segmentation - Segment objects by referring expressions",
    "img_reaseg": "Image reasoning segmentation - Segment objects by reasoning questions",
    "img_gcgseg": "Image GCG segmentation - Generate caption then segment objects in the caption",
    "img_intseg": "Interactive segmentation - Segment objects by the interactive prompt",
    "img_vgdseg": "Image VGD segmentation - Segment objects by the visual grounded prompt",
}

SUPPORTED_TASKS = list(TASK_DESCRIPTION.keys())

EXAMPLES = {
    "img_chat": [
        osp.join(this_dir, "./sample.jpg"),
        "What is unusal about this image?",
        "img_chat",
    ],
    "img_genseg": [
        osp.join(this_dir, "./sample.jpg"),
        "ins: "
        + ", ".join([c["name"] for c in COCO_INSTANCE_CATEGORIES])
        + ";\nsem: "
        + ", ".join([c["name"] for c in COCO_SEMANTIC_CATEGORIES]),
        "img_genseg",
    ],
    "img_refseg": [
        osp.join(this_dir, "./sample.jpg"),
        "the ironing man",
        "img_refseg",
    ],
    "img_reaseg": [
        osp.join(this_dir, "./sample.jpg"),
        "What can be used to warm clothes?",
        "img_reaseg",
    ],
    "img_gcgseg": [
        osp.join(this_dir, "./sample.jpg"),
        "Can you provide a brief description of this image? Please respond with interleaved segmentation masks for the corresponding phrases.",
        "img_gcgseg",
    ],
    "img_intseg": [
        osp.join(this_dir, "./sample.jpg"),
        "You DON'T NEED to input any prompt for img_intseg. Draw the object you want to segment on the image.",
        "img_intseg",
    ],
    "img_vgdseg": [
        osp.join(this_dir, "./sample.jpg"),
        "You DON'T NEED to input any prompt for img_vgdseg. Draw the object you want to segment on the image.",
        "img_vgdseg",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="X-SAM Gradio Demo")
    parser.add_argument("config", help="config file name or path")
    parser.add_argument("--work-dir", help="directory to save logs and visualizations")
    parser.add_argument("--pth_model", type=str, default="latest", help="path to model checkpoint or 'latest'")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--log-dir", type=str, default="./logs", help="directory to save logs")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override config options")
    parser.add_argument("--port", type=int, default=7860, help="port for gradio server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="host for gradio server")
    parser.add_argument("--share", action="store_true", help="share gradio app")
    return parser.parse_args()


class GradioApp:
    def __init__(self, demo: XSamDemo, log_dir: str):
        self.demo = demo
        self.log_dir = osp.abspath(log_dir)
        self.processing_status = "Ready"

    def _need_visual_prompt(self, task_name):
        return task_name in ["img_intseg", "img_vgdseg"]

    def _empty_image_editor(self):
        return {"background": None, "layers": [], "composite": None}

    def _build_image_editor_value(self, background=None, layers=None):
        if background is None:
            return self._empty_image_editor()
        background = load_image(background, mode="RGB").convert("RGBA")
        normalized_layers = []
        for layer in layers or []:
            if layer is None:
                continue
            normalized_layers.append(load_image(layer, mode="RGB").convert("RGBA"))
        return {"background": background, "layers": normalized_layers, "composite": background}

    def _load_image_vprompt(self, image_data):
        if image_data is None:
            return self._empty_image_editor()
        return self._build_image_editor_value(image_data)

    def _parse_image_editor_data(self, data):
        if not isinstance(data, dict):
            return None
        if data.get("background") is None:
            return None

        vprompt_masks = [np.array(layer)[..., -1] for layer in (data.get("layers") or []) if layer is not None]
        vprompt_masks = [mask for mask in vprompt_masks if mask.sum() > 0]
        return vprompt_masks or None

    def _get_vprompt_update(self, task_name, image_data=None):
        if not self._need_visual_prompt(task_name):
            return gr.update(value=self._empty_image_editor(), visible=False)
        if image_data is None:
            return gr.update(value=self._empty_image_editor(), visible=False)
        return gr.update(value=self._load_image_vprompt(image_data), visible=True)

    def _on_image_change(self, image_data, task_name):
        status_message = (
            "📸 Image uploaded! Draw visual prompts, then click 'Run X-SAM'."
            if image_data is not None and self._need_visual_prompt(task_name)
            else (
                "📸 Image uploaded! Enter a prompt and click 'Run X-SAM'."
                if image_data is not None
                else "🟢 Ready to process - Upload an image and enter a prompt!"
            )
        )
        return self._get_vprompt_update(task_name, image_data=image_data), status_message

    def _update_task_ui(self, task_name, image_data=None):
        vprompt_input_update = self._get_vprompt_update(task_name, image_data=image_data)
        status_message = (
            "🟢 Ready to process - Upload an image to start drawing visual prompts!"
            if self._need_visual_prompt(task_name)
            else "🟢 Ready to process - Upload an image and enter a prompt!"
        )
        return (
            TASK_DESCRIPTION.get(task_name, ""),
            vprompt_input_update,
            gr.update(value=None, visible=True, height=480),
            status_message,
        )

    def gradio_predict_with_progress(
        self,
        image_data,
        vprompt_data,
        prompt,
        task_name="img_chat",
        score_thr=0.5,
        progress=gr.Progress(),
    ):
        if image_data is None:
            return (
                "❌ No image provided",
                "",
                "",
                gr.update(value=None, height=480),
            )

        try:
            progress(0.1, desc="🔍 Initializing...")

            if not prompt or prompt.strip() == "":
                if task_name not in [
                    "img_gcgseg",
                    "img_intseg",
                    "img_vgdseg",
                ]:
                    return "❌ No prompt provided", "", "", gr.update(value=None, height=480)

            day_timestamp = datetime.datetime.now().strftime("%Y%m%d")
            file_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            day_log_dir = osp.join(self.log_dir, day_timestamp)
            log_file = osp.join(day_log_dir, f"{day_timestamp}.log")
            out_log_dir = osp.join(day_log_dir, "output")

            os.makedirs(day_log_dir, exist_ok=True)
            os.makedirs(out_log_dir, exist_ok=True)

            progress(0.3, desc="🖼️ Processing image...")
            progress(0.5, desc="🔎 Running X-SAM...")
            start_time = time.time()
            vprompt_masks = self._parse_image_editor_data(vprompt_data)
            llm_input, llm_output, image_output = self.demo.run_on_image(
                image_data,
                prompt,
                task_name,
                vprompt_masks=vprompt_masks,
                threshold=score_thr,
                output_dir=out_log_dir,
                file_prefix=file_timestamp,
            )

            llm_success = llm_output is not None
            seg_success = image_output is not None
            inference_time = time.time() - start_time

            progress(0.9, desc="💾 Saving results...")
            if not osp.exists(log_file):
                with open(log_file, "w") as f:
                    f.write("timestamp\tinput\tprompt\ttask_name\tinference_time\tllm_success\tseg_success\n")
            with open(log_file, "a") as f:
                f.write(
                    f"{file_timestamp}\t{file_timestamp}.png\t{prompt}\t{task_name}\t{inference_time:.3f}\t{llm_success}\t{seg_success}\n"
                )

            progress(1.0, desc="✅ Complete!")

            last_error = getattr(self.demo, "_last_infer_error", None)
            if llm_success or seg_success:
                status_message = f"✅ Completed successfully in {inference_time:.2f}s."
            elif XSamDemo._is_cuda_oom(last_error):
                status_message = (
                    f"❌ GPU out of memory ({inference_time:.2f}s). "
                    "Use a smaller image."
                )
            else:
                status_message = f"⚠️ Failed in {inference_time:.2f}s."
            self.demo._last_infer_error = None

            return (
                status_message,
                llm_input,
                llm_output,
                (
                    gr.update(value=image_output, height=image_output.shape[0] + 10, visible=True)
                    if image_output is not None
                    else gr.update(value=None, visible=True, height=480)
                ),
            )

        except Exception as e:
            print(f"Error in gradio_predict: {traceback.format_exc()}")
            XSamDemo._release_gpu_memory(aggressive=XSamDemo._is_cuda_oom(e))
            if XSamDemo._is_cuda_oom(e):
                error_msg = (
                    "❌ GPU out of memory. Use a smaller image."
                    "If it still fails, please restart the demo process."
                )
            else:
                error_msg = f"❌ Error: {str(e)}"
            if hasattr(self.demo, "_last_infer_error"):
                self.demo._last_infer_error = None
            return error_msg, "", "", gr.update(value=None, height=480)

    def create_interface(self):
        examples = []
        for _, example_data in EXAMPLES.items():
            if example_data and len(example_data) >= 3:
                data_path, text_prompt, task_name = example_data
                if not data_path:
                    continue
                if osp.isfile(data_path):
                    try:
                        image = load_image(data_path)
                        examples.append(
                            [
                                image,
                                (
                                    self._load_image_vprompt(image)
                                    if self._need_visual_prompt(task_name)
                                    else self._empty_image_editor()
                                ),
                                text_prompt,
                                task_name,
                                0.5,
                            ]
                        )
                    except Exception as e:
                        print(f"Error loading example image {data_path}: {e}")
                        continue

        # Theme aligned with the project webpage.
        theme = gr.themes.Soft(
            primary_hue="emerald",
            secondary_hue="sky",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "system-ui", "sans-serif"],
            radius_size=gr.themes.sizes.radius_lg,
            text_size=gr.themes.sizes.text_md,
        ).set(
            body_background_fill="*neutral_50",
            block_background_fill="#ffffff",
            block_border_width="1px",
            block_border_color="#e2e8f0",
            button_large_radius="999px",
        )

        with gr.Blocks(title="X-SAM", css=custom_css, theme=theme) as app:
            # Header
            # <a href="https://arxiv.org/abs/2603.00000" target="_blank" rel="noopener noreferrer">
            #     <img class="link-icon" src="https://cdn.simpleicons.org/arxiv/B31B1B" alt="arXiv">
            #     <span>arXiv</span>
            # </a>
            gr.HTML(
                """
                <div class="main-header">
                    <div class="main-header-inner">
                        <h1 class="publication-title">✨ X-SAM ✨</h1>
                        <h2 class="publication-subtitle">Any Segmentation in Images</h2>
                        <div class="header-links">
                            <a href="https://wanghao9610.github.io/X-SAM/" target="_blank" rel="noopener noreferrer">
                                <span class="link-emoji" aria-hidden="true">🌐</span>
                                <span>Project</span>
                            </a>
                            <a href="https://huggingface.co/hao9610/X-SAM/blob/main/X-SAM.pdf" target="_blank" rel="noopener noreferrer">
                                <span class="link-emoji" aria-hidden="true">📄</span>
                                <span>Paper</span>
                            </a>
                            <a href="https://huggingface.co/hao9610/X-SAM" target="_blank" rel="noopener noreferrer">
                                <img class="link-icon" src="https://cdn.simpleicons.org/huggingface/FF9D00" alt="HuggingFace">
                                <span>HuggingFace</span>
                            </a>
                            <a href="https://github.com/wanghao9610/X-SAM" target="_blank" rel="noopener noreferrer">
                                <img class="link-icon" src="https://cdn.simpleicons.org/github/181717" alt="GitHub">
                                <span>Code</span>
                            </a>
                        </div>
                    </div>
                </div>
                """
            )

            # Main Content
            with gr.Row(elem_classes="main-row"):

                # ================= Left Column: Inputs =================
                with gr.Column(scale=5, elem_classes="input-section"):
                    with gr.Group(elem_classes="media-card"):
                        image_input = gr.Image(
                            type="pil",
                            label="📸 Image",
                            elem_classes="image-upload",
                            sources=["upload", "webcam", "clipboard"],
                            height=480,
                        )
                        vprompt_input = gr.ImageEditor(
                            type="pil",
                            label="🖌️ Visual Prompt",
                            elem_classes="image-upload",
                            brush=gr.Brush(
                                colors=["#FF0000", "#00FF00", "#0000FF", "#FF00FF", "#00FFFF"],
                                default_color="#FF0000",
                                default_size=5,
                            ),
                            eraser=gr.Eraser(default_size=5),
                            visible=False,
                        )

                    with gr.Group(elem_classes="controls-card"):
                        with gr.Row(elem_classes="task-row"):
                            task_name = gr.Dropdown(
                                choices=SUPPORTED_TASKS,
                                value="img_chat",
                                label="🎯 Task",
                                scale=1,
                                elem_classes="task-select",
                            )
                            task_description = gr.Textbox(
                                value=TASK_DESCRIPTION["img_chat"],
                                label="📋 Description",
                                interactive=False,
                                lines=2,
                                scale=1,
                                elem_classes="description-box",
                            )
                        with gr.Row():
                            suggestions_btn = gr.Button(
                                "💡 Load Example",
                                size="sm",
                                elem_classes="example-btn",
                            )

                        with gr.Group(elem_classes="prompt-group"):
                            text_input = gr.Textbox(
                                lines=3,
                                max_lines=20,
                                label="🤔 Prompt",
                                placeholder="Enter your prompt here based on the task selected...",
                                elem_classes="auto-resize-textbox",
                            )

                        score_thr = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0.5,
                            step=0.01,
                            interactive=True,
                            label="🔍 Score Threshold",
                        )

                    with gr.Row(elem_classes="action-buttons"):
                        clear_btn = gr.Button("🗑️ Clear All", variant="secondary", size="lg")
                        submit_btn = gr.Button("🚀 Run X-SAM", variant="primary", size="lg", elem_classes="run-btn")

                # ================= Right Column: Outputs =================
                with gr.Column(scale=6, elem_classes="output-section"):
                    status_display = gr.Textbox(
                        value="🟢 Ready to process - Upload an image and enter a prompt!",
                        label="ℹ️ Status",
                        interactive=False,
                        elem_classes="running-info",
                        lines=1,
                        max_lines=3,
                    )

                    with gr.Group(elem_classes="panel-section mask-panel"):
                        gr.Markdown(
                            "### <span class='section-title'><span class='icon'>📷</span> Segmentation Mask</span>"
                        )
                        image_output = gr.Image(
                            type="pil",
                            label="Image Output",
                            show_label=False,
                            height=480,
                            elem_classes="mask-media",
                        )

                    with gr.Group(elem_classes="panel-section"):
                        gr.Markdown("### <span class='section-title'><span class='icon'>💬</span> Conversation</span>")
                        llm_input = gr.Textbox(
                            label="📝 Instruction",
                            placeholder="Parsed language instruction will appear here...",
                            lines=3,
                            max_lines=10,
                            interactive=False,
                            elem_classes="auto-resize-textbox",
                        )
                        llm_output = gr.Textbox(
                            label="🤖 Response",
                            placeholder="Model response will appear here...",
                            lines=3,
                            max_lines=10,
                            interactive=False,
                            elem_classes="auto-resize-textbox",
                        )

            # Examples
            if examples:
                with gr.Group(elem_classes="examples-section"):
                    gr.Markdown("### 🌟 Example")
                    gr.Examples(
                        examples=examples,
                        inputs=[image_input, vprompt_input, text_input, task_name, score_thr],
                        outputs=[status_display, llm_input, llm_output, image_output],
                        fn=self.gradio_predict_with_progress,
                        cache_examples=False,
                        examples_per_page=20,
                    )

            # Event Handlers
            submit_btn.click(
                fn=self.gradio_predict_with_progress,
                inputs=[image_input, vprompt_input, text_input, task_name, score_thr],
                outputs=[status_display, llm_input, llm_output, image_output],
                show_progress=True,
            )

            clear_btn.click(
                fn=lambda: [
                    gr.update(value=None, visible=True),
                    gr.update(value={"background": None, "layers": [], "composite": None}, visible=False),
                    "",
                    "img_chat",
                    "",
                    "",
                    gr.update(value=None, visible=True, height=480),
                    0.5,
                    "🧹 All cleared! Ready for new input.",
                ],
                outputs=[
                    image_input,
                    vprompt_input,
                    text_input,
                    task_name,
                    llm_input,
                    llm_output,
                    image_output,
                    score_thr,
                    status_display,
                ],
            )

            suggestions_btn.click(
                fn=self.get_examples,
                inputs=[task_name],
                outputs=[image_input, vprompt_input, text_input],
            )

            task_name.change(
                fn=self._update_task_ui,
                inputs=[task_name, image_input],
                outputs=[
                    task_description,
                    vprompt_input,
                    image_output,
                    status_display,
                ],
            )

            image_input.change(
                fn=self._on_image_change,
                inputs=[image_input, task_name],
                outputs=[vprompt_input, status_display],
            )

        return app

    def get_examples(self, task_name):
        example = EXAMPLES.get(task_name, None)
        if not example:
            return None, gr.update(value=self._empty_image_editor(), visible=False), ""
        try:
            data_path, text_prompt = example[0], example[1]
            if data_path and osp.isfile(data_path):
                image = Image.open(data_path).convert("RGB")
                return (
                    image,
                    gr.update(
                        value=(
                            self._load_image_vprompt(image)
                            if self._need_visual_prompt(task_name)
                            else self._empty_image_editor()
                        ),
                        visible=self._need_visual_prompt(task_name),
                    ),
                    text_prompt,
                )
            return None, gr.update(value=self._empty_image_editor(), visible=False), text_prompt
        except Exception as e:
            print(f"Error processing example: {e}")
            return None, gr.update(value=self._empty_image_editor(), visible=False), ""


def setup_cfg(args):
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
    args = parse_args()
    args, cfg = setup_cfg(args)
    args.log_dir = osp.abspath(args.log_dir)
    os.makedirs(args.log_dir, exist_ok=True)

    print_log("Initializing X-SAM demo...", logger="current")
    demo = XSamDemo(cfg, args.pth_model, output_ids_with_output=False)
    print_log("X-SAM demo initialized successfully!", logger="current")

    gradio_app = GradioApp(demo, args.log_dir)
    app = gradio_app.create_interface()

    print_log(f"Starting Gradio server on {args.host}:{args.port}", logger="current")
    app.launch(
        show_error=True,
        share=args.share,
        server_port=args.port,
        server_name=args.host,
        allowed_paths=[args.log_dir],
    )


if __name__ == "__main__":
    main()

import os
import os.path as osp
import re
import string

import pandas as pd
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from PIL import Image
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionModel,
    GenerationConfig,
    SiglipImageProcessor,
    SiglipVisionModel,
    StoppingCriteriaList,
)
from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp import *
from vlmeval.vlm.base import BaseModel

from xsam.dataset.processors import Qwen3VLImageProcessor, Sam2ImageProcessor, SamImageProcessor
from xsam.dataset.utils.image import expand2square
from xsam.model.segmentors.sam import SamModel
from xsam.model.segmentors.sam2 import Sam2Model
from xsam.model.utils import prepare_inputs_labels_for_mlm
from xsam.model.vlms.qwen3vl import Qwen3VLForConditionalGeneration
from xsam.utils.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_PLACEHOLDER_TOKEN, TOKEN2INDEX
from xsam.utils.criteria import StopWordStoppingCriteria
from xsam.utils.logging import print_log
from xsam.utils.template import PROMPT_TEMPLATE


class XSamChat(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = False

    def __init__(
        self,
        xsam_path,
        llm_path=None,
        vlm_path=None,
        segmentor_path=None,
        vision_encoder_path=None,
        segmention_encoder_path=None,
        visual_select_layer=-2,
        visual_select_indx=0,
        prompt_template=None,
        image_token=None,
        stop_words=[],
        torch_dtype=torch.float16,
        expand2square=False,
        use_placeholder=False,
        use_dual_encoder=False,
    ):
        if not osp.isdir(xsam_path):
            cache_path = get_cache_path(xsam_path)
            if cache_path is not None:
                xsam_path = cache_path
            else:
                xsam_path = snapshot_download(repo_id=xsam_path)
        assert osp.exists(xsam_path) and osp.isdir(xsam_path)

        vlm = None
        llm = None
        image_processor = None
        extra_image_processor = None
        vision_encoder = None
        segmention_encoder = None
        visual_projector = None
        segmentor_projector = None

        # build vlm
        if vlm_path is not None:
            vlm = Qwen3VLForConditionalGeneration.from_pretrained(
                vlm_path, trust_remote_code=False, low_cpu_mem_usage=True, torch_dtype=torch_dtype, device_map="cpu"
            )
            image_processor = Qwen3VLImageProcessor.from_pretrained(vlm_path)
            tokenizer = AutoTokenizer.from_pretrained(
                vlm_path, trust_remote_code=False, low_cpu_mem_usage=True, encode_special_tokens=True
            )
            print_log(f"Load vlm from {vlm_path}", logger="current")

            token_num, token_dim = vlm.lm_head.out_features, vlm.lm_head.in_features
            if vlm.lm_head.weight.shape[0] != token_num:
                vlm.lm_head.weight = torch.nn.Parameter(
                    torch.empty(token_num, token_dim, device=vlm.device, dtype=vlm.dtype)
                )
                vlm.model.embed_tokens.weight = torch.nn.Parameter(
                    torch.empty(token_num, token_dim, device=vlm.device, dtype=vlm.dtype)
                )
        elif "vlm" in os.listdir(xsam_path):
            vlm_path = osp.join(xsam_path, "vlm")
            vlm = Qwen3VLForConditionalGeneration.from_pretrained(
                vlm_path, trust_remote_code=False, low_cpu_mem_usage=True, torch_dtype=torch_dtype, device_map="cpu"
            )
            image_processor = Qwen3VLImageProcessor.from_pretrained(vlm_path)
            tokenizer = AutoTokenizer.from_pretrained(
                vlm_path, trust_remote_code=False, low_cpu_mem_usage=True, encode_special_tokens=True
            )
            print_log(f"Load vlm from {vlm_path}", logger="current")

            token_num, token_dim = vlm.lm_head.out_features, vlm.lm_head.in_features
            if vlm.lm_head.weight.shape[0] != token_num:
                vlm.lm_head.weight = torch.nn.Parameter(
                    torch.empty(token_num, token_dim, device=vlm.device, dtype=vlm.dtype)
                )
                vlm.model.embed_tokens.weight = torch.nn.Parameter(
                    torch.empty(token_num, token_dim, device=vlm.device, dtype=vlm.dtype)
                )

        # build llm
        if llm_path is not None:
            llm = AutoModelForCausalLM.from_pretrained(
                llm_path, trust_remote_code=False, low_cpu_mem_usage=True, torch_dtype=torch_dtype, device_map="cpu"
            )
            tokenizer = AutoTokenizer.from_pretrained(
                llm_path, trust_remote_code=False, low_cpu_mem_usage=True, encode_special_tokens=True
            )
            print_log(f"Load llm from {llm_path}", logger="current")

            token_num, token_dim = llm.lm_head.out_features, llm.lm_head.in_features
            if llm.lm_head.weight.shape[0] != token_num:
                llm.lm_head.weight = torch.nn.Parameter(
                    torch.empty(token_num, token_dim, device=llm.device, dtype=llm.dtype)
                )
                llm.model.embed_tokens.weight = torch.nn.Parameter(
                    torch.empty(token_num, token_dim, device=llm.device, dtype=llm.dtype)
                )
        elif "llm" in os.listdir(xsam_path):
            assert llm_path is None, "Please don't specify the `llm_path` since passed " "`xsam_path` contains a llm!"
            llm_path = osp.join(xsam_path, "llm")
            llm = AutoModelForCausalLM.from_pretrained(
                llm_path, trust_remote_code=False, low_cpu_mem_usage=True, torch_dtype=torch_dtype, device_map="cpu"
            )
            tokenizer = AutoTokenizer.from_pretrained(
                llm_path, trust_remote_code=False, low_cpu_mem_usage=True, encode_special_tokens=True
            )
            print_log(f"Load llm from {llm_path}", logger="current")

            token_num, token_dim = llm.lm_head.out_features, llm.lm_head.in_features
            if llm.lm_head.weight.shape[0] != token_num:
                llm.lm_head.weight = torch.nn.Parameter(
                    torch.empty(token_num, token_dim, device=llm.device, dtype=llm.dtype)
                )
                llm.model.embed_tokens.weight = torch.nn.Parameter(
                    torch.empty(token_num, token_dim, device=llm.device, dtype=llm.dtype)
                )

        # build vision_encoder
        if "vision_encoder" in os.listdir(xsam_path):
            assert vision_encoder_path is None, (
                "Please don't specify the `vision_encoder_path` since passed "
                "`xsam_path` contains a visual encoder!"
            )
            vision_encoder_path = osp.join(xsam_path, "vision_encoder")

            if "clip" in vision_encoder_path:
                vision_encoder = CLIPVisionModel.from_pretrained(
                    vision_encoder_path, torch_dtype=torch_dtype, device_map="cpu"
                )
                image_processor = CLIPImageProcessor.from_pretrained(vision_encoder_path)
            elif "siglip" in vision_encoder_path:
                vision_encoder = SiglipVisionModel.from_pretrained(
                    vision_encoder_path, torch_dtype=torch_dtype, device_map="cpu"
                )
                image_processor = SiglipImageProcessor.from_pretrained(vision_encoder_path)
            else:
                raise ValueError(f"Unsupported visual encoder: {vision_encoder_path}")
            print_log(f"Load vision_encoder from {vision_encoder_path}", logger="current")

        # build segmention_encoder
        if use_dual_encoder and "segmention_encoder" in os.listdir(xsam_path):
            assert segmention_encoder_path is None, (
                "Please don't specify the `segmention_encoder_path` since passed "
                "`xsam_path` contains a segmentor encoder!"
            )
            segmention_encoder_path = osp.join(xsam_path, "segmention_encoder")

            segmention_encoder = Sam2Model.from_pretrained(
                segmention_encoder_path, torch_dtype=torch_dtype, device_map="cpu"
            )
            segmention_encoder = segmention_encoder.vision_encoder
            extra_image_processor = Sam2ImageProcessor.from_pretrained(segmention_encoder_path)
            print_log(f"Load segmention_encoder from {segmention_encoder_path}", logger="current")
        elif use_dual_encoder and segmentor_path is not None:
            if "sam" in segmentor_path:
                segmentor = SamModel.from_pretrained(segmentor_path, torch_dtype=torch_dtype, device_map="cpu")
                segmention_encoder = segmentor.vision_encoder
                extra_image_processor = SamImageProcessor.from_pretrained(segmentor_path)
            elif "sam2" in segmentor_path:
                segmentor = Sam2Model.from_pretrained(segmentor_path, torch_dtype=torch_dtype, device_map="cpu")
                segmention_encoder = segmentor.vision_encoder
                extra_image_processor = Sam2ImageProcessor.from_pretrained(segmentor_path)
            else:
                raise ValueError(f"Unsupported segmentor: {segmentor_path}")
            print_log(f"Load segmentor from {segmentor_path}", logger="current")

        # load adapter
        if "llm_adapter" in os.listdir(xsam_path):
            adapter_path = osp.join(xsam_path, "llm_adapter")
            llm = PeftModel.from_pretrained(
                llm,
                adapter_path,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
                device_map="cpu",
            )
            print_log(f"Load llm adapter from {adapter_path}", logger="current")

        if "vlm_adapter" in os.listdir(xsam_path):
            adapter_path = osp.join(xsam_path, "vlm_adapter")
            vlm = PeftModel.from_pretrained(
                vlm,
                adapter_path,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
                device_map="cpu",
            )
            print_log(f"Load vlm adapter from {adapter_path}", logger="current")

        if "vision_encoder_adapter" in os.listdir(xsam_path):
            adapter_path = osp.join(xsam_path, "vision_encoder_adapter")
            vision_encoder = PeftModel.from_pretrained(
                vision_encoder,
                adapter_path,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
                device_map="cpu",
            )
            print_log(f"Load vision_encoder adapter from {adapter_path}", logger="current")

        # TODO: add segmention_encoder_adapter
        if "segmention_encoder_adapter" in os.listdir(xsam_path):
            adapter_path = osp.join(xsam_path, "segmention_encoder_adapter")
            segmention_encoder = PeftModel.from_pretrained(
                segmention_encoder,
                adapter_path,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
                device_map="cpu",
            )
            print_log(f"Load segmention_encoder adapter from {adapter_path}", logger="current")

        # build visual_projector
        if "visual_projector" in os.listdir(xsam_path):
            visual_projector_path = osp.join(xsam_path, "visual_projector")
            visual_projector = AutoModel.from_pretrained(
                visual_projector_path,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
                torch_dtype=torch_dtype,
                device_map="cpu",
            )
            print_log(f"Load visual_projector from {visual_projector_path}", logger="current")

        # build segmentor_projector
        if "segmentor_projector" in os.listdir(xsam_path):
            segmentor_projector_path = osp.join(xsam_path, "segmentor_projector")
            segmentor_projector = AutoModel.from_pretrained(
                segmentor_projector_path,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
                torch_dtype=torch_dtype,
                device_map="cpu",
            )
            print_log(f"Load segmentor_projector from {segmentor_projector_path}", logger="current")

        mlm = llm or vlm
        mlm.eval()
        self.mlm = mlm.cuda()

        if vision_encoder is not None:
            vision_encoder.eval()
            visual_projector.eval()
            self.vision_encoder = vision_encoder.cuda()
            self.visual_projector = visual_projector.cuda()

        if segmention_encoder is not None:
            segmention_encoder.eval()
            segmentor_projector.eval()
            self.segmention_encoder = segmention_encoder.cuda()
            self.segmentor_projector = segmentor_projector.cuda()

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.extra_image_processor = extra_image_processor
        self.visual_select_layer = visual_select_layer
        self.visual_select_indx = visual_select_indx

        self.prompt_template = PROMPT_TEMPLATE.get(prompt_template, None)
        assert self.prompt_template is not None, f"Unsupported prompt template: {prompt_template}"
        stop_words += self.prompt_template.get("STOP_WORDS", [])

        self.stop_criteria = StoppingCriteriaList()
        for word in stop_words:
            self.stop_criteria.append(StopWordStoppingCriteria(self.tokenizer, word))

        self.image_token = image_token
        self.expand2square = expand2square
        self.use_placeholder = use_placeholder
        self.use_dual_encoder = use_dual_encoder

    def build_gen_config(self, dataset_name):
        gen_kwargs = dict(
            max_new_tokens=2048,
            do_sample=True,
            temperature=1,
            num_beams=5,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=(
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            ),
        )
        # For single word generation
        if dataset_name is not None and DATASET_TYPE(dataset_name) in [
            "MCQ",
            "Y/N",
        ]:
            gen_kwargs.update(dict(max_new_tokens=16, do_sample=False, num_beams=1))
        return GenerationConfig(**gen_kwargs)

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == "MCQ":
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line["question"]
        hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
        if hint is not None:
            question = hint + "\n" + question

        options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
        for key, item in options.items():
            question += f"\n{key}. {item}"

        if not cn_string(question):
            prompt = question + "\n" + ("Answer with the option's letter " "from the given choices directly.")
        else:
            prompt = question + "\n" + "请直接回答选项字母。"

        message = [dict(type="text", value=prompt)]
        message.extend([dict(type="image", value=s) for s in tgt_path])
        return message

    def generate_inner(self, message, dataset=None):
        pixel_values = None
        image_embeds = None
        image_grid_thw = None
        deepstack_image_embeds = None

        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        prompt = prompt.replace("<image>", "")
        pil_image = Image.open(image_path).convert("RGB")
        images = pil_image
        if self.expand2square:
            images = expand2square(pil_image, tuple(int(x * 255) for x in self.image_processor.image_mean))
        outputs = self.image_processor.preprocess(images, return_tensors="pt")
        pixel_values = outputs["pixel_values"] if outputs["pixel_values"].ndim == 4 else outputs["pixel_values"]
        pixel_values = pixel_values.to(self.mlm.device, self.mlm.dtype)
        image_grid_thw = outputs.get("image_grid_thw", None)
        image_grid_thw = image_grid_thw.to(self.mlm.device) if image_grid_thw is not None else None

        extend_mm_inputs = {}
        mm_inputs = {}
        if self.mlm is not None:
            if pixel_values is None:
                raise ValueError("No pixel values provided")

            image_embeds, deepstack_image_embeds = self.mlm.get_image_features(pixel_values, image_grid_thw)
            extend_mm_inputs.update(
                {
                    "image_embeds": torch.cat(image_embeds, dim=0).to(self.mlm.device, self.mlm.dtype),
                    "image_grid_thw": image_grid_thw,
                    "deepstack_image_embeds": deepstack_image_embeds,
                }
            )
            mm_inputs["pixel_values"] = image_embeds

        # llm + vision_encoder
        if getattr(self, "vision_encoder", None) is not None:
            visual_outputs = self.vision_encoder(pixel_values, output_hidden_states=True)
            pixel_values = self.visual_projector(
                visual_outputs.hidden_states[self.visual_select_layer][:, self.visual_select_indx :]
            )
            mm_inputs["pixel_values"] = pixel_values

        if (
            self.use_dual_encoder
            and getattr(self, "segmention_encoder", None) is not None
            and getattr(self, "segmentor_projector", None) is not None
        ):
            extra_pixel_values = self.extra_image_processor.preprocess(pil_image, return_tensors="pt")["pixel_values"]
            extra_pixel_values = extra_pixel_values.to(self.mlm.device, self.mlm.dtype)
            seg_outputs = self.segmention_encoder(extra_pixel_values, output_hidden_states=True)
            extra_pixel_values = self.segmentor_projector(seg_outputs.hidden_states[self.visual_select_layer])
        else:
            extra_pixel_values = None
        mm_inputs["extra_pixel_values"] = extra_pixel_values

        inputs = (self.image_token or f"{DEFAULT_IMAGE_TOKEN}\n") + prompt

        if image_grid_thw is not None and self.use_placeholder:
            merge_length = getattr(self.image_processor, "merge_size", 1) ** 2
            index = 0
            while (self.image_token or DEFAULT_IMAGE_TOKEN) in inputs:
                num_image_tokens = image_grid_thw[index].prod() // merge_length
                inputs = inputs.replace(DEFAULT_IMAGE_TOKEN, "<|placeholder|>" * num_image_tokens, 1)
                index += 1
            inputs = inputs.replace("<|placeholder|>", DEFAULT_PLACEHOLDER_TOKEN)

        if self.prompt_template:
            system = (
                self.prompt_template["SYSTEM"].format(system=prompt.get("system", ""))
                if isinstance(prompt, dict) and "system" in prompt
                else ""
            )
            prefix = prompt.get("assistant", "") if isinstance(prompt, dict) and "assistant" in prompt else ""
            inputs = system + self.prompt_template["INSTRUCTION"].format(input=inputs) + prefix

        input_ids = []
        pattern = f"({'|'.join(re.escape(token) for token in TOKEN2INDEX.keys())})"
        chunks = [chunk for chunk in re.split(pattern, inputs) if chunk.strip() != ""]
        input_ids = [
            (
                [TOKEN2INDEX[chunk]]
                if chunk in TOKEN2INDEX
                else self.tokenizer.encode(chunk, add_special_tokens=(idx == 0))
            )
            for idx, chunk in enumerate(chunks)
        ]
        input_ids = [id for sublist in input_ids if isinstance(sublist, list) for id in sublist]
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.mlm.device)
        mm_inputs["input_ids"] = input_ids

        mm_inputs = prepare_inputs_labels_for_mlm(
            mlm=self.mlm,
            use_dual_encoder=self.use_dual_encoder,
            **mm_inputs,
        )
        mm_inputs.update(extend_mm_inputs)
        if mm_inputs.get("input_ids", None) is not None:
            mm_inputs["input_ids"] = None

        gen_config = self.build_gen_config(dataset.dataset_name)
        generate_output = self.mlm.generate(
            **mm_inputs,
            streamer=None,
            bos_token_id=self.tokenizer.bos_token_id,
            generation_config=gen_config,
            stopping_criteria=self.stop_criteria,
        )
        predict = self.tokenizer.decode(generate_output[0], skip_special_tokens=True).strip()
        return predict

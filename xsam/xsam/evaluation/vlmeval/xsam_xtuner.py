import os
import os.path as osp
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
from xtuner.utils import StopWordStoppingCriteria

from xsam.dataset.processors import SamImageProcessor
from xsam.model.segmentors.sam import SamModel
from xsam.utils.logging import print_log
from xsam.utils.template import PROMPT_TEMPLATE


class XSam_XTuner(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = False

    def __init__(
        self,
        xsam_path,
        llm_path=None,
        segmentor_path=None,
        visual_encoder_path=None,
        segmentor_encoder_path=None,
        visual_select_layer=-2,
        visual_select_indx=0,
        prompt_template=None,
        stop_words=[],
        torch_dtype=torch.float16,
    ):
        if not osp.isdir(xsam_path):
            cache_path = get_cache_path(xsam_path)
            if cache_path is not None:
                xsam_path = cache_path
            else:
                xsam_path = snapshot_download(repo_id=xsam_path)
        assert osp.exists(xsam_path) and osp.isdir(xsam_path)

        # build visual_encoder
        if "llm" in os.listdir(xsam_path):
            assert llm_path is None, "Please don't specify the `llm_path` since passed " "`xsam_path` contains a LLM!"
            llm_path = osp.join(xsam_path, "llm")
        else:
            assert llm_path is not None, "Please specify the `llm_path`!"

        llm = AutoModelForCausalLM.from_pretrained(
            llm_path, trust_remote_code=False, low_cpu_mem_usage=True, torch_dtype=torch_dtype, device_map="cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            llm_path, trust_remote_code=False, low_cpu_mem_usage=True, encode_special_tokens=True
        )
        print_log(f"Load LLM from {llm_path}", logger="current")

        token_num, token_dim = llm.lm_head.out_features, llm.lm_head.in_features
        if llm.lm_head.weight.shape[0] != token_num:
            llm.lm_head.weight = torch.nn.Parameter(
                torch.empty(token_num, token_dim, device=llm.device, dtype=llm.dtype)
            )
            llm.model.embed_tokens.weight = torch.nn.Parameter(
                torch.empty(token_num, token_dim, device=llm.device, dtype=llm.dtype)
            )

        # build visual_encoder
        if "visual_encoder" in os.listdir(xsam_path):
            assert visual_encoder_path is None, (
                "Please don't specify the `visual_encoder_path` since passed " "`xsam_path` contains a visual encoder!"
            )
            visual_encoder_path = osp.join(xsam_path, "visual_encoder")
        else:
            assert visual_encoder_path is not None, "Please specify the `visual_encoder_path`!"

        if "clip" in visual_encoder_path:
            visual_encoder = CLIPVisionModel.from_pretrained(
                visual_encoder_path, torch_dtype=torch_dtype, device_map="cpu"
            )
            image_processor = CLIPImageProcessor.from_pretrained(visual_encoder_path)
        elif "siglip" in visual_encoder_path:
            visual_encoder = SiglipVisionModel.from_pretrained(
                visual_encoder_path, torch_dtype=torch_dtype, device_map="cpu"
            )
            image_processor = SiglipImageProcessor.from_pretrained(visual_encoder_path)
        else:
            raise ValueError(f"Unsupported visual encoder: {visual_encoder_path}")
        print_log(f"Load visual_encoder from {visual_encoder_path}", logger="current")

        # build segmentor_encoder
        if "segmentor_encoder" in os.listdir(xsam_path):
            assert segmentor_encoder_path is None, (
                "Please don't specify the `segmentor_encoder_path` since passed "
                "`xsam_path` contains a segmentor encoder!"
            )
            segmentor_encoder_path = osp.join(xsam_path, "segmentor_encoder")

        if segmentor_path is not None and "sam" in segmentor_path:
            segmentor = SamModel.from_pretrained(segmentor_path, torch_dtype=torch_dtype, device_map="cpu")
            segmentor_encoder = segmentor.vision_encoder
            extra_image_processor = SamImageProcessor.from_pretrained(segmentor_path)
            print_log(f"Load segmentor from {segmentor_path}", logger="current")
        elif segmentor_encoder_path is not None:
            segmentor_encoder = SamModel.from_pretrained(
                segmentor_encoder_path, torch_dtype=torch_dtype, device_map="cpu"
            )
            segmentor_encoder = segmentor_encoder.vision_encoder
            extra_image_processor = SamImageProcessor.from_pretrained(segmentor_encoder_path)
            print_log(f"Load segmentor_encoder from {segmentor_encoder_path}", logger="current")
        else:
            segmentor_encoder = None
            extra_image_processor = None

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
            print_log(f"Load LLM adapter from {adapter_path}", logger="current")

        if "visual_encoder_adapter" in os.listdir(xsam_path):
            adapter_path = osp.join(xsam_path, "visual_encoder_adapter")
            visual_encoder = PeftModel.from_pretrained(
                visual_encoder,
                adapter_path,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
                device_map="cpu",
            )
            print_log(f"Load visual_encoder adapter from {adapter_path}", logger="current")

        # TODO: add segmentor_encoder_adapter
        if "segmentor_encoder_adapter" in os.listdir(xsam_path):
            adapter_path = osp.join(xsam_path, "segmentor_encoder_adapter")
            segmentor_encoder = PeftModel.from_pretrained(
                segmentor_encoder,
                adapter_path,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
                device_map="cpu",
            )
            print_log(f"Load segmentor_encoder adapter from {adapter_path}", logger="current")

        # build visual_projector
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
        segmentor_projector_path = osp.join(xsam_path, "segmentor_projector")
        if osp.exists(segmentor_projector_path):
            segmentor_projector = AutoModel.from_pretrained(
                segmentor_projector_path,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
                torch_dtype=torch_dtype,
                device_map="cpu",
            )
            print_log(f"Load segmentor_projector from {segmentor_projector_path}", logger="current")
        else:
            segmentor_projector = None

        llm.eval()
        visual_encoder.eval()
        visual_projector.eval()

        self.llm = llm.cuda()
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.visual_encoder = visual_encoder.cuda()
        self.visual_projector = visual_projector.cuda()
        self.extra_image_processor = extra_image_processor
        self.segmentor_encoder = segmentor_encoder.cuda() if segmentor_encoder is not None else None
        self.segmentor_projector = segmentor_projector.cuda() if segmentor_projector is not None else None
        self.visual_select_layer = visual_select_layer
        self.visual_select_indx = visual_select_indx
        self.prompt_template = PROMPT_TEMPLATE.get(prompt_template, None)
        assert self.prompt_template is not None, f"Unsupported prompt template: {prompt_template}"
        stop_words += self.prompt_template.get("STOP_WORDS", [])

        self.stop_criteria = StoppingCriteriaList()
        for word in stop_words:
            self.stop_criteria.append(StopWordStoppingCriteria(self.tokenizer, word))

    def build_gen_config(self, dataset):
        gen_kwargs = dict(
            max_new_tokens=512,
            do_sample=True,
            temperature=1,
            num_beams=5,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=(
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            ),
        )
        # For single word generation
        if dataset is not None and DATASET_TYPE(dataset) in ["MCQ", "Y/N"]:
            gen_kwargs.update(dict(max_new_tokens=5, do_sample=False, num_beams=1))
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
        from xtuner.dataset.utils import expand2square
        from xtuner.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX

        from xsam.model.utils import prepare_inputs_labels_for_multimodal

        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        prompt = prompt.replace("<image>", "")
        pil_image = Image.open(image_path).convert("RGB")
        pixel_values = expand2square(pil_image, tuple(int(x * 255) for x in self.image_processor.image_mean))
        pixel_values = self.image_processor.preprocess(pixel_values, return_tensors="pt")["pixel_values"][0]
        pixel_values = pixel_values.cuda().unsqueeze(0)
        visual_outputs = self.visual_encoder(pixel_values, output_hidden_states=True)
        pixel_values = self.visual_projector(
            visual_outputs.hidden_states[self.visual_select_layer][:, self.visual_select_indx :]
        )
        if self.segmentor_projector is not None:
            extra_pixel_values = self.extra_image_processor.preprocess(pil_image, return_tensors="pt")["pixel_values"][
                0
            ]
            extra_pixel_values = extra_pixel_values.cuda().unsqueeze(0)
            seg_outputs = self.segmentor_encoder(extra_pixel_values, output_hidden_states=True)
            extra_pixel_values = self.segmentor_projector(seg_outputs.hidden_states[self.visual_select_layer])
        else:
            extra_pixel_values = None
        inputs = DEFAULT_IMAGE_TOKEN + "\n" + prompt

        if self.prompt_template:
            inputs = self.prompt_template["INSTRUCTION"].format(input=inputs)

        chunk_encode = []
        for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
            if idx == 0:
                cur_encode = self.tokenizer(chunk)
            else:
                cur_encode = self.tokenizer(chunk, add_special_tokens=False)
            chunk_encode.append(cur_encode)
        assert len(chunk_encode) == 2
        ids = []
        for idx, cur_chunk_encode in enumerate(chunk_encode):
            ids.extend(cur_chunk_encode["input_ids"])
            if idx != len(chunk_encode) - 1:
                ids.append(IMAGE_TOKEN_INDEX)
        ids = torch.tensor(ids).cuda().unsqueeze(0)
        mm_inputs = prepare_inputs_labels_for_multimodal(
            llm=self.llm, input_ids=ids, pixel_values=pixel_values, extra_pixel_values=extra_pixel_values
        )
        if mm_inputs.get("input_ids", None) is not None:
            mm_inputs["input_ids"] = None

        gen_config = self.build_gen_config(dataset)
        generate_output = self.llm.generate(
            **mm_inputs,
            generation_config=gen_config,
            streamer=None,
            bos_token_id=self.tokenizer.bos_token_id,
            stopping_criteria=self.stop_criteria,
        )
        predict = self.tokenizer.decode(generate_output[0], skip_special_tokens=True).strip()
        return predict

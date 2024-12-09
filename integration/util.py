from typing import Tuple
import re
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, AddedToken
from minicons import scorer

from integration.abstract import Object


class Util(Object):
    pass


def load_model_and_processor(
    model_name: str, padding_side: str = "left"
) -> Tuple[AutoModelForVision2Seq, AutoProcessor]:
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        load_in_8bit=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained(model_name, padding_side=padding_side)

    if "blip2" in model_name and "coco" in model_name:
        processor.num_query_tokens = model.config.num_query_tokens
        image_token = AddedToken("<image>", normalized=False, special=True)
        processor.tokenizer.add_tokens([image_token], special_tokens=True)
        model.resize_token_embeddings(len(processor.tokenizer), pad_to_multiple_of=64)
        model.config.image_token_index = len(processor.tokenizer) - 1
    Util.info("Loaded " + model_name)
    return model, processor


def load_scorer(model_name: str) -> scorer.VLMScorer:
    vlm_scorer = scorer.VLMScorer(
        model_name, 
        "auto",
        load_in_8bit=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )
    if "blip2" in model_name and "coco" in model_name:
        vlm_scorer.tokenizer.num_query_tokens = vlm_scorer.model.config.num_query_tokens
        image_token = AddedToken("<image>", normalized=False, special=True)
        vlm_scorer.tokenizer.tokenizer.add_tokens([image_token], special_tokens=True)
        vlm_scorer.model.resize_token_embeddings(len(vlm_scorer.tokenizer.tokenizer), pad_to_multiple_of=64)
        vlm_scorer.model.config.image_token_index = len(vlm_scorer.tokenizer.tokenizer) - 1
    Util.info("Loaded " + model_name + " as scorer")
    return vlm_scorer


def to_scorer(model: AutoModelForVision2Seq, processor: AutoProcessor) -> scorer.VLMScorer:
    processor.vocab_size = processor.tokenizer.vocab_size
    vlm_scorer = scorer.VLMScorer(model, 'auto', processor)
    return vlm_scorer


def format_prompt(prompt: str, model_name: str) -> str:
    if "google" in model_name:
        return "<image><bos>" + prompt
    if "kosmos" in model_name:
        return "<grounding> " + prompt
    return prompt


def parse_decoded_output(prompt: str, decoded_output: str) -> str:
    # remove tags from kosmos and google
    decoded_output = re.sub("<.+?>", "", decoded_output)
    # remove prompt from those that include it
    if prompt in decoded_output:
        decoded_output = decoded_output[decoded_output.index(prompt) + len(prompt) :]
    # remove leading/trailing whitespace
    decoded_output = decoded_output.strip()
    return decoded_output

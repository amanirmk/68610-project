from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import copy
import re
import random
import pandas as pd
import numpy as np
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    AdamW,
)
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from integration.util import (
    parse_decoded_output,
    load_model_and_processor,
    format_prompt,
)
from integration.abstract import Object


class VQA(Object):
    pass


# ADAPTED FROM https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py

CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}
MANUAL_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
ARTICLES = ["a", "an", "the"]
PERIOD_STRIP = re.compile("(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile("(\d)(\,)(\d)")
PUNCT = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]


def processPunctuation(inText: str) -> str:
    outText = inText
    for p in PUNCT:
        if (p + " " in inText or " " + p in inText) or (
            re.search(COMMA_STRIP, inText) != None
        ):
            outText = outText.replace(p, "")
        else:
            outText = outText.replace(p, " ")
    outText = PERIOD_STRIP.sub("", outText, re.UNICODE)
    return outText


def processDigitArticle(inText: str) -> str:
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = MANUAL_MAP.setdefault(word, word)
        if word not in ARTICLES:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in CONTRACTIONS:
            outText[wordId] = CONTRACTIONS[word]
    outTextStr = " ".join(outText)
    return outTextStr


def eval_vqa(results: List[Dict[str, Any]]) -> Tuple[float, float]:
    question_exact_accs = []
    question_partial_accs = []
    for question in tqdm(results, desc="Analyzing VQA Predictions"):
        for real_answer in question["answers"]:
            real_answer["answer"] = real_answer["answer"].replace("\n", " ")
            real_answer["answer"] = real_answer["answer"].replace("\t", " ")
            real_answer["answer"] = real_answer["answer"].strip()
            real_answer["answer"] = processPunctuation(real_answer["answer"])
            real_answer["answer"] = processDigitArticle(real_answer["answer"])
        model_answer = question["answer"]
        model_answer = model_answer.replace("\n", " ")
        model_answer = model_answer.replace("\t", " ")
        model_answer = model_answer.strip()
        model_answer = processPunctuation(model_answer)
        model_answer = processDigitArticle(model_answer)

        exact_answer_accs = []
        partial_answer_accs = []
        for real_answer in question["answers"]:
            other_answers = [
                other_answer
                for other_answer in question["answers"]
                if other_answer != real_answer
            ]
            exact_matching_answers = [
                other_answer
                for other_answer in other_answers
                if other_answer["answer"] == model_answer
            ]
            partial_matching_answers = [
                other_answer
                for other_answer in other_answers
                if other_answer["answer"] and other_answer["answer"] in model_answer
            ]
            exact_acc = min(1, float(len(exact_matching_answers)) / 3)
            partial_acc = min(1, float(len(partial_matching_answers)) / 3)
            exact_answer_accs.append(exact_acc)
            partial_answer_accs.append(partial_acc)
        avg_exact_answer_acc = float(sum(exact_answer_accs)) / len(exact_answer_accs)
        avg_partial_answer_acc = float(sum(partial_answer_accs)) / len(
            partial_answer_accs
        )
        question_exact_accs.append(avg_exact_answer_acc)
        question_partial_accs.append(avg_partial_answer_acc)

    overall_exact_acc = float(sum(question_exact_accs)) / len(question_exact_accs)
    overall_partial_acc = float(sum(question_partial_accs)) / len(question_partial_accs)
    VQA.info(
        f"Exact acc = {overall_exact_acc:.3f}, Partial acc = {overall_partial_acc:.3f}"
    )
    return overall_exact_acc, overall_partial_acc


# ORIGINAL CODE


def zeroshot_vqa2(
    model_name: str,
    save_intermediate: bool = False,
    limit: Optional[int] = None,
) -> Tuple[float, float]:
    model, processor = load_model_and_processor(model_name, padding_side="left")
    return test_vqa2(
        model_name,
        model,
        processor,
        save_intermediate=save_intermediate,
        limit=limit,
        tag="zeroshot",
    )


def test_vqa2(
    model_name: str,
    model: AutoModelForVision2Seq,
    processor: AutoProcessor,
    save_intermediate: bool = False,
    max_answer_tokens: int = 30,
    batch_size: int = 250,
    limit: Optional[int] = None,
    tag: Optional[str] = None,
) -> Tuple[float, float]:
    model.eval()
    processor.tokenizer.padding_side = "left"

    dataset = load_dataset(
        "HuggingFaceM4/VQAv2", split="validation", trust_remote_code=True
    )
    if limit is not None:
        dataset = dataset.select(range(limit))
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x
    )

    results = []

    for batch in tqdm(dataloader, desc="Collecting VQA Predictions"):
        questions_batch = [
            format_prompt(
                f"Answer the following question about the image.\nQuestion: {example['question']}\nAnswer:",
                model_name,
            )
            for example in batch
        ]
        images_batch = [example["image"] for example in batch]
        results_batch = [
            {
                "question_id": example["question_id"],
                "answers": copy.deepcopy(example["answers"]),
            }
            for example in batch
        ]

        inputs = processor(
            images=images_batch, text=questions_batch, return_tensors="pt", padding=True
        ).to("cuda", torch.float16)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_answer_tokens)

        for i in range(len(results_batch)):
            results_batch[i]["answer"] = parse_decoded_output(
                questions_batch[i],
                processor.tokenizer.decode(output_ids[i], skip_special_tokens=True),
            )
        results.extend(results_batch)

    if save_intermediate:
        pd.DataFrame(results)[["question_id", "answer"]].to_csv(
            f"{model_name.replace('/', '--')}-vqa2-{tag + '-' if tag else ''}answers.csv",
            index=False,
        )
    return eval_vqa(results)


def finetune_vqa2(
    model_name: str,
    save_intermediate: bool = False,
    num_epochs: int = 1,
    batch_size: int = 16,
    learning_rate: float = 1e-7,
    train_limit: Optional[int] = None,
    eval_limit: Optional[int] = None,
    gradient_accumulation_steps: int = 4,
    warmup_steps: int = 100,
) -> Tuple[float, float]:
    model, processor = load_model_and_processor(model_name)

    VQA.info("Loading training data")
    train_dataset = load_dataset("HuggingFaceM4/VQAv2", split="train")
    VQA.info("Training data loaded")
    if train_limit is not None:
        train_dataset = train_dataset.select(range(train_limit))

    VQA.info(
        f"Training on {len(train_dataset)} examples for {num_epochs} epoch(s) in batches of {batch_size}."
    )

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x
    )

    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = len(train_dataloader) * num_epochs
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Training loop
    model.train()
    decoder_mask_works = True
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for step, batch in enumerate(progress_bar):
            questions_batch = [
                format_prompt(
                    f"Answer the following question about the image.\nQuestion: {example['question'].capitalize().rstrip('?') + '?'}\nAnswer:",
                    model_name,
                )
                for example in batch
            ]
            images_batch = [example["image"] for example in batch]
            answers_batch = [example["multiple_choice_answer"] for example in batch]

            # Process inputs
            processor.tokenizer.padding_side = "left"  # keep text tokens recent
            inputs = processor(
                images=images_batch,
                text=questions_batch,
                return_tensors="pt",
                padding=True,
            ).to("cuda", torch.float16)

            # Process targets
            processor.tokenizer.padding_side = (
                "right"  # immediately predict text tokens
            )
            targets = processor.tokenizer(  # use tokenizer so no image tokens added
                text=answers_batch,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,  # don't start with </s>
            ).to("cuda")

            target_ids = torch.where(
                targets.input_ids == processor.tokenizer.pad_token_id,
                torch.tensor(-100),
                targets.input_ids,
            )

            # Forward pass
            if decoder_mask_works:
                try:
                    outputs = model(
                        **inputs,
                        labels=targets.input_ids,
                        decoder_attention_mask=targets.attention_mask,
                    )
                except Exception as e:
                    if "decoder_attention_mask" in str(e):
                        decoder_mask_works = False
                        # -100 ids are ignored by torch cross entropy loss
                        target_ids = torch.where(
                            targets.input_ids == processor.tokenizer.pad_token_id,
                            torch.tensor(-100),
                            targets.input_ids,
                        )
                        outputs = model(
                            **inputs,
                            labels=target_ids,
                        )
                    else:
                        raise e
            else:
                target_ids = torch.where(
                    targets.input_ids == processor.tokenizer.pad_token_id,
                    torch.tensor(-100),
                    targets.input_ids,
                )
                outputs = model(
                    **inputs,
                    labels=target_ids,
                )

            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()

            # Gradient accumulation
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps
            progress_bar.set_postfix(
                {"loss": loss.item() * gradient_accumulation_steps}
            )

        avg_loss = total_loss / len(train_dataloader)
        VQA.info(f"Average loss: {avg_loss:.4f} at epoch {epoch+1}")

    if save_intermediate:
        output_dir = f"{'finetuned/' + model_name.replace('/', '--')}-vqa2-finetuned"
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)

    return test_vqa2(
        model_name,
        model,
        processor,
        save_intermediate=save_intermediate,
        limit=eval_limit,
        tag="finetune",
    )

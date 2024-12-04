from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import copy
import re
import pandas as pd
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

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
                if other_answer["answer"] in model_answer
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
    return overall_exact_acc, overall_partial_acc


# ORIGINAL CODE


def zeroshot_vqa2(
    model_name: str,
    device: str,
    max_answer_tokens: int = 10,
    batch_size: int = 250,
    limit: Optional[int] = None,
) -> Tuple[float, float]:
    model = AutoModelForVision2Seq.from_pretrained(model_name).to(device)
    processor = AutoProcessor.from_pretrained(model_name, padding_side="left")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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
            "QUESTION: " + example["question"] + "\nANSWER: " for example in batch
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
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_answer_tokens)

        for i in range(len(results_batch)):
            results_batch[i]["answer"] = tokenizer.decode(
                output_ids[i], skip_special_tokens=True
            )
        results.extend(results_batch)

    pd.DataFrame(results)[["question_id", "answer"]].to_csv(
        f"{model_name.replace('/', '--')}-vqa2-answers.csv", index=False
    )
    return eval_vqa(results)

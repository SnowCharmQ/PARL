import os
import math
import json
import asyncio
import logging
import argparse
from tqdm import tqdm
from openai import AsyncOpenAI
from transformers import set_seed
from datasets import load_dataset, load_from_disk, concatenate_datasets

from prompt import REVIEW_RUBRICS_EVALUATOR_SYSTEM_PROMPT, REVIEW_RUBRICS_EVALUATOR_USER_PROMPT_TEMPLATE

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
set_seed(42)

METHODS = [
    "GT",
    "NON", "RAG", "PAG", "NON_THINK", "RAG_THINK", "PAG_THINK", "R2P",
    "SFT", "NEXTQUILL",
    "GRPO", "GSPO", "PRLM",
    "TagPR", "SFT_GRPO",
]

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--non_path", type=str, required=True)
parser.add_argument("--rag_path", type=str, required=True)
parser.add_argument("--pag_path", type=str, required=True)
parser.add_argument("--non_think_path", type=str, required=True)
parser.add_argument("--rag_think_path", type=str, required=True)
parser.add_argument("--pag_think_path", type=str, required=True)
parser.add_argument("--r2p_path", type=str, required=True)
parser.add_argument("--sft_path", type=str, required=True)
parser.add_argument("--nextquill_path", type=str, required=True)
parser.add_argument("--grpo_path", type=str, required=True)
parser.add_argument("--gspo_path", type=str, required=True)
parser.add_argument("--prlm_path", type=str, required=True)
parser.add_argument("--tagpr_path", type=str, required=True)
parser.add_argument("--sft_grpo_path", type=str, required=True)
args = parser.parse_args()

data_dir = "outputs/" + args.data_dir

rubrics_dataset = load_from_disk(data_dir)
rubrics_dataset = {f"{item['user_id']}-{item['category']}": item['rubrics'] for item in rubrics_dataset}

model_name = "Qwen/Qwen3-235B-A22B-Instruct-2507"
client = AsyncOpenAI(api_key=os.getenv("RUBRICS_API_KEY"), base_url=os.getenv("RUBRICS_BASE_URL"))
structured_outputs_params = {"structured_outputs": {"choice": ["yes", "no"]}}

movies_dataset = load_dataset(
    "SnowCharmQ/DPL-main",
    "Movies_and_TV",
    split="test"
).map(lambda _: {"category": "Movies_and_TV"})
books_dataset = load_dataset(
    "SnowCharmQ/DPL-main",
    "Books",
    split="test"
).map(lambda _: {"category": "Books"})
main_dataset = concatenate_datasets([movies_dataset, books_dataset])
movies_meta_dataset = load_dataset(
    "SnowCharmQ/DPL-meta",
    "Movies_and_TV",
    split="full"
)
books_meta_dataset = load_dataset(
    "SnowCharmQ/DPL-meta",
    "Books",
    split="full"
)
meta_dataset = concatenate_datasets([movies_meta_dataset, books_meta_dataset])
meta_dataset = dict(
    zip(
        meta_dataset["asin"],
        zip(meta_dataset["title"], meta_dataset["description"])
    )
)

SEM = asyncio.Semaphore(80)

def postprocess_output(output):
    idx = output.find("</think>")
    if idx != -1:
        output = output[idx + len("</think>"):].strip()
    idx = output.find("[Review]:")
    if idx != -1:
        output = output[idx + len("[Review]:"):].strip()
    idx = output.find("[Review Text]:")
    if idx != -1:
        output = output[idx + len("[Review Text]:"):].strip()
    idx = output.find("[Headline]:")
    if idx != -1:
        output = output[idx + len("[Headline]:"):].strip()
    idx = output.find("[Content]:")
    if idx != -1:
        output = output[idx + len("[Content]:"):].strip()
    return output

def load_results(movies_file_path, books_file_path):
    with open(movies_file_path, "r") as f:
        movies_predictions = f.read()
        movies_predictions = movies_predictions.split('\n---------------------------------\n')
        movies_predictions = movies_predictions[:-1]
        movies_predictions = [pred.strip() for pred in movies_predictions]
    with open(books_file_path, "r") as f:
        books_predictions = f.read()
        books_predictions = books_predictions.split('\n---------------------------------\n')
        books_predictions = books_predictions[:-1]
        books_predictions = [pred.strip() for pred in books_predictions]
    predictions = movies_predictions + books_predictions
    return predictions

def load_non_results():
    movies_file_path = f"{args.non_path}/predictions_Movies_and_TV.txt"
    books_file_path = f"{args.non_path}/predictions_Books.txt"
    return load_results(movies_file_path, books_file_path)
NON_RESULTS = load_non_results()

def load_rag_results():
    movies_file_path = f"{args.rag_path}/predictions_Movies_and_TV.txt"
    books_file_path = f"{args.rag_path}/predictions_Books.txt"
    return load_results(movies_file_path, books_file_path)
RAG_RESULTS = load_rag_results()

def load_pag_results():
    movies_file_path = f"{args.pag_path}/predictions_Movies_and_TV.txt"
    books_file_path = f"{args.pag_path}/predictions_Books.txt"
    return load_results(movies_file_path, books_file_path)
PAG_RESULTS = load_pag_results()

def load_non_think_results():
    movies_file_path = f"{args.non_think_path}/predictions_Movies_and_TV.txt"
    books_file_path = f"{args.non_think_path}/predictions_Books.txt"
    return load_results(movies_file_path, books_file_path)
NON_THINK_RESULTS = load_non_think_results()

def load_rag_think_results():
    movies_file_path = f"{args.rag_think_path}/predictions_Movies_and_TV.txt"
    books_file_path = f"{args.rag_think_path}/predictions_Books.txt"
    return load_results(movies_file_path, books_file_path)
RAG_THINK_RESULTS = load_rag_think_results()

def load_pag_think_results():
    movies_file_path = f"{args.pag_think_path}/predictions_Movies_and_TV.txt"
    books_file_path = f"{args.pag_think_path}/predictions_Books.txt"
    return load_results(movies_file_path, books_file_path)
PAG_THINK_RESULTS = load_pag_think_results()

def load_r2p_results():
    movies_file_path = f"{args.r2p_path}/predictions_Movies_and_TV.txt"
    books_file_path = f"{args.r2p_path}/predictions_Books.txt"
    return load_results(movies_file_path, books_file_path)
R2P_RESULTS = load_r2p_results()

def load_sft_results():
    movies_file_path = f"{args.sft_path}/predictions_Movies_and_TV.txt"
    books_file_path = f"{args.sft_path}/predictions_Books.txt"
    return load_results(movies_file_path, books_file_path)
SFT_RESULTS = load_sft_results()

def load_nextquill_results():
    movies_file_path = f"{args.nextquill_path}/predictions_Movies_and_TV.txt"
    books_file_path = f"{args.nextquill_path}/predictions_Books.txt"
    return load_results(movies_file_path, books_file_path)
NEXTQUILL_RESULTS = load_nextquill_results()

def load_grpo_results():
    movies_file_path = f"{args.grpo_path}/predictions_Movies_and_TV.txt"
    books_file_path = f"{args.grpo_path}/predictions_Books.txt"
    return load_results(movies_file_path, books_file_path)
GRPO_RESULTS = load_grpo_results()

def load_gspo_results():
    movies_file_path = f"{args.gspo_path}/predictions_Movies_and_TV.txt"
    books_file_path = f"{args.gspo_path}/predictions_Books.txt"
    return load_results(movies_file_path, books_file_path)
GSPO_RESULTS = load_gspo_results()

def load_prlm_results():
    movies_file_path = f"{args.prlm_path}/predictions_Movies_and_TV.txt"
    books_file_path = f"{args.prlm_path}/predictions_Books.txt"
    return load_results(movies_file_path, books_file_path)
PRLM_RESULTS = load_prlm_results()

def load_tagpr_results():
    movies_file_path = f"{args.tagpr_path}/predictions_Movies_and_TV.txt"
    books_file_path = f"{args.tagpr_path}/predictions_Books.txt"
    return load_results(movies_file_path, books_file_path)
TAGPR_RESULTS = load_tagpr_results()

def load_sft_grpo_results():
    movies_file_path = f"{args.sft_grpo_path}/predictions_Movies_and_TV.txt"
    books_file_path = f"{args.sft_grpo_path}/predictions_Books.txt"
    return load_results(movies_file_path, books_file_path)
SFT_GRPO_RESULTS = load_sft_grpo_results()

def build_rubrics_evaluate_prompt(meta_info, response, rubric):
    user_prompt = REVIEW_RUBRICS_EVALUATOR_USER_PROMPT_TEMPLATE.format(
        metadata=meta_info, 
        response=response, 
        rubric=rubric,
    )
    return [
        {"role": "system", "content": REVIEW_RUBRICS_EVALUATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

async def evaluate_score(prompt):
    while True:
        try:
            completion = await client.chat.completions.create(
                model=model_name,
                messages=prompt,
                max_tokens=1,
                logprobs=True,
                top_logprobs=5,
                extra_body=structured_outputs_params,
            )
            logprobs_data = completion.choices[0].logprobs.content[0].top_logprobs
            token_probs = {lp.token.lower().strip(): lp.logprob for lp in logprobs_data}
            l_yes = token_probs.get("yes", -20)
            l_no = token_probs.get("no", -20)
            max_l = max(l_yes, l_no)
            p_yes = math.exp(l_yes - max_l) / (math.exp(l_yes - max_l) + math.exp(l_no - max_l))
            return p_yes
        except Exception as e:
            await asyncio.sleep(1)
            continue

async def process_sample(sample, review):
    async with SEM:
        data = sample["data"]
        user_id = sample["user_id"]
        category = sample["category"]
        rubrics = rubrics_dataset[f"{user_id}-{category}"]
        data = sample['data']
        asin = data['asin']
        item_title, item_description = meta_dataset[asin]
        review_rating = data['rating']
        review_title = data['title']
        meta_info = (
            f"[Item Title]: {item_title}\n"
            f"[Item Description]: {item_description}\n"
            f"[Output Review Rating]: {review_rating}\n"
            f"[Output Review Title]: {review_title}\n"
        )
        if len(rubrics) == 0:
            return None
        scores = []
        for rubric in rubrics:
            prompt = build_rubrics_evaluate_prompt(meta_info, review, rubric)
            score = await evaluate_score(prompt)
            if score > 0.5:
                scores.append(score)
        return 1 if len(scores) == len(rubrics) else 0

async def main():
    rubrics_score_results = {}
    for method in METHODS:
        if method == "GT":
            pass
        elif method == "NON":
            results = NON_RESULTS
        elif method == "RAG":
            results = RAG_RESULTS
        elif method == "PAG":
            results = PAG_RESULTS
        elif method == "NON_THINK":
            results = NON_THINK_RESULTS
        elif method == "RAG_THINK":
            results = RAG_THINK_RESULTS
        elif method == "PAG_THINK":
            results = PAG_THINK_RESULTS
        elif method == "R2P":
            results = R2P_RESULTS
        elif method == "SFT":
            results = SFT_RESULTS
        elif method == "NEXTQUILL":
            results = NEXTQUILL_RESULTS
        elif method == "GRPO":
            results = GRPO_RESULTS
        elif method == "GSPO":
            results = GSPO_RESULTS
        elif method == "PRLM":
            results = PRLM_RESULTS
        elif method == "TagPR":
            results = TAGPR_RESULTS
        elif method == "SFT_GRPO":
            results = SFT_GRPO_RESULTS
        tasks = []
        for idx, sample in enumerate(main_dataset):
            review = results[idx] if method != "GT" else sample['data']['text']
            review = postprocess_output(review)
            task = process_sample(sample, review)
            tasks.append(task)
        scores = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing dataset"):
            score = await coro
            scores.append(score)
        valid_scores = [score for score in scores if score is not None]
        score = sum(valid_scores) / len(valid_scores)
        rubrics_score_results[method] = score
        print(method, score)

    with open(f"{data_dir}/rubrics_scores.json", "w") as f:
        json.dump(rubrics_score_results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    asyncio.run(main())

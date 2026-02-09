import os
import math
import json
import asyncio
import logging
import argparse
import warnings
from openai import AsyncOpenAI
from transformers import set_seed
from datasets import load_from_disk
from tqdm.asyncio import tqdm as tqdm_asyncio

from prompt import REVIEW_RUBRICS_EVALUATOR_SYSTEM_PROMPT, REVIEW_RUBRICS_EVALUATOR_USER_PROMPT_TEMPLATE
from prompt import REVIEW_RUBRICS_GENERATOR_SYSTEM_PROMPT, REVIEW_RUBRICS_GENERATOR_USER_PROMPT_TEMPLATE

warnings.filterwarnings("ignore")
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default="0,1,2,3,4,5,6,7")

args = parser.parse_args()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

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

model_generator = "qwen3-parl-b"
client_generator = AsyncOpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
model_evaluator = "Qwen/Qwen3-235B-A22B-Instruct-2507"
client_evaluator = AsyncOpenAI(api_key=os.getenv("RUBRICS_API_KEY"), base_url=os.getenv("RUBRICS_BASE_URL"))
structured_outputs_params = {"structured_outputs": {"choice": ["yes", "no"]}}
SEM = asyncio.Semaphore(80)

main_dataset = load_from_disk("../../datasets/amazon")

async def evaluate_score(prompt):
    while True:
        try:
            completion = await client_evaluator.chat.completions.create(
                model=model_evaluator,
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
            print(f"Error evaluating score: {e}")
            await asyncio.sleep(1)
            continue

async def generate_rubrics(sample):
    metadata = sample['metadata']
    profile = metadata['profile']
    num = min(len(profile), 10)
    profile_score_infos = [
        ((
            f"[Item Title]: {prof['item_title']}\n"
            f"[Item Description]: {prof['item_description']}\n"
            f"[Output Review Rating]: {prof['review_rating']}\n"
            f"[Output Review Title]: {prof['review_title']}\n"
        ), prof['review_text']) for prof in profile
    ]
    async with SEM:
        best_rubrics = []
        cnt = 0
        while True:
            cnt += 1
            profile = profile[-num:]
            past_reviews = "\n".join([
                f"{profile[i]['review_text']}\n"
                for i in range(num)
            ])
            rubrics_generator_prompt = [
                {
                    "role": "system",
                    "content": REVIEW_RUBRICS_GENERATOR_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": REVIEW_RUBRICS_GENERATOR_USER_PROMPT_TEMPLATE.format(
                        historical_reviews=past_reviews
                    )
                }
            ]
            try:
                result = await client_generator.chat.completions.create(
                    model=model_generator,
                    messages=rubrics_generator_prompt,
                )
                content = result.choices[0].message.content
                content = postprocess_output(content)
                rubrics = json.loads(content)
                valid_rubrics = []
                for rubric in rubrics:
                    rubric = rubric['rule']
                    scores = []
                    for metainfo, gt in profile_score_infos:
                        gt = postprocess_output(gt)
                        gt_evaluate_prompt = build_rubrics_evaluate_prompt(metainfo, gt, rubric)
                        gt_score = await evaluate_score(gt_evaluate_prompt)
                        if gt_score <= 0.5:
                            break
                        scores.append(gt_score)
                    if len(scores) < len(profile_score_infos):
                        continue
                    valid_rubrics.append(rubric)
                if len(best_rubrics) == 0 or len(valid_rubrics) > len(best_rubrics):
                    best_rubrics = valid_rubrics
                if cnt >= 5:
                    break
            except Exception as e:
                print(f"Error generating rubrics: {e}")
                message = str(e)
                if message.find("This model's maximum context length is 32768 tokens.") != -1:
                    num -= 1
                await asyncio.sleep(1)
                continue
        print(len(best_rubrics))
        return best_rubrics

async def main():
    tasks = [generate_rubrics(sample) for sample in main_dataset]
    results = await tqdm_asyncio.gather(*tasks, desc="Generating rubrics")

    def add_rubrics(example, idx):
        example["rubrics"] = results[idx]
        return example

    dataset = main_dataset.map(add_rubrics, with_indices=True, desc="Adding rubrics to dataset")
    dataset.save_to_disk("outputs/amazon_parl_b")

if __name__ == "__main__":
    asyncio.run(main())

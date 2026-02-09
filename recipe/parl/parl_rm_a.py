import os
import sys
import json
import math
import torch
import asyncio
import threading
import numpy as np
from openai import AsyncOpenAI
from concurrent.futures import ThreadPoolExecutor

_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)
from prompt import REVIEW_RUBRICS_EVALUATOR_SYSTEM_PROMPT, REVIEW_RUBRICS_EVALUATOR_USER_PROMPT_TEMPLATE
from prompt import NEWS_RUBRICS_EVALUATOR_SYSTEM_PROMPT, NEWS_RUBRICS_EVALUATOR_USER_PROMPT_TEMPLATE
from prompt import REDDIT_RUBRICS_EVALUATOR_SYSTEM_PROMPT, REDDIT_RUBRICS_EVALUATOR_USER_PROMPT_TEMPLATE

SEMAPHORE_SIZE = 80
NUM_WORKERS = min(os.cpu_count(), 80)
GLOBAL_SEM = threading.Semaphore(SEMAPHORE_SIZE)
RUBRICS_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
RUBRICS_BASE_URL = os.getenv("RUBRICS_BASE_URL")
RUBRICS_API_KEY = os.getenv("RUBRICS_API_KEY")
RUBRICS_STRUCTURED_OUTPUTS_PARAMS = {"structured_outputs": {"choice": ["yes", "no"]}}

def postprocess(output):
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

def build_rubrics_evaluate_prompt(data_source, metadata, response, rubric):
    if data_source == "amazon_rubrics":
        system_prompt = REVIEW_RUBRICS_EVALUATOR_SYSTEM_PROMPT
        user_prompt = REVIEW_RUBRICS_EVALUATOR_USER_PROMPT_TEMPLATE.format(
            metadata=metadata, 
            response=response, 
            rubric=rubric,
        )
    elif data_source == "news_rubrics":
        system_prompt = NEWS_RUBRICS_EVALUATOR_SYSTEM_PROMPT
        user_prompt = NEWS_RUBRICS_EVALUATOR_USER_PROMPT_TEMPLATE.format(
            content=metadata, 
            response=response, 
            rubric=rubric,
        )
    elif data_source == "reddit_rubrics":
        system_prompt = REDDIT_RUBRICS_EVALUATOR_SYSTEM_PROMPT
        user_prompt = REDDIT_RUBRICS_EVALUATOR_USER_PROMPT_TEMPLATE.format(
            summary=metadata, 
            content=response, 
            rubric=rubric,
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

async def evaluate_score(client, prompt):
    while True:
        try:
            completion = await client.chat.completions.create(
                model=RUBRICS_MODEL,
                messages=prompt,
                max_tokens=1,
                logprobs=True,
                top_logprobs=5,
                extra_body=RUBRICS_STRUCTURED_OUTPUTS_PARAMS,
            )
            logprobs_data = completion.choices[0].logprobs.content[0].top_logprobs
            token_probs = {lp.token.lower().strip(): lp.logprob for lp in logprobs_data}
            l_yes = token_probs.get("yes", -20)
            l_no = token_probs.get("no", -20)
            max_l = max(l_yes, l_no)
            p_yes = math.exp(l_yes - max_l) / (math.exp(l_yes - max_l) + math.exp(l_no - max_l))
            p_no = math.exp(l_no - max_l) / (math.exp(l_yes - max_l) + math.exp(l_no - max_l))
            return p_yes
        except Exception as e:
            print(f"Error evaluating score: {e}")
            await asyncio.sleep(1)
            continue

async def process_sample_rubrics(args):
    data_source, metadata, rubrics, gens, profile = args
    gt, pd_non, pd_rag, pd_sft, pd_grpo, pd_sft_grpo = gens
    try:
        rubrics = json.loads(rubrics)
        if len(rubrics) < 2 or len(rubrics) > 10:
            raise Exception(f"rubrics length is not in [2, 10]")
        for rubric in rubrics:
            if rubric.get("rule") is None:
                raise Exception(f"rubric rule is None")
    except Exception as e:
        print(f"Error parsing rubrics: {e}")
        return {
            "gt_score": 0.0,
            "pd_non_score": 0.0,
            "pd_rag_score": 0.0,
            "pd_sft_score": 0.0,
            "pd_grpo_score": 0.0,
            "pd_sft_grpo_score": 0.0,
            "total_rubrics": 0,
            "valid_rubrics": 0,
        }
    if data_source == "amazon_rubrics":
        profile_score_infos = [
            ((
                f"[Item Title]: {prof['item_title']}\n"
                f"[Item Description]: {prof['item_description']}\n"
                f"[Output Review Rating]: {prof['review_rating']}\n"
                f"[Output Review Title]: {prof['review_title']}\n"
            ), prof['review_text']) for prof in profile
        ]
    elif data_source == "lamp4_rubrics":
        profile_score_infos = [
            (prof['text'], prof['title']) for prof in profile
        ]
    elif data_source == "topic_rubrics":
        profile_score_infos = [
            (prof['summary'], prof['content']) for prof in profile
        ]
    sem = asyncio.Semaphore(SEMAPHORE_SIZE)
    async with AsyncOpenAI(base_url=RUBRICS_BASE_URL, api_key=RUBRICS_API_KEY) as client:
        async with sem:
            valid_rubrics = []
            for rubric in rubrics:
                rubric = rubric.get("rule")
                if rubric is None:
                    continue
                if len(rubric) < 20 or len(rubric) > 400:
                    continue
                scores = []
                for metainfo, gt in profile_score_infos:
                    gt_evaluate_prompt = build_rubrics_evaluate_prompt(data_source, metainfo, gt, rubric)
                    gt_score = await evaluate_score(client, gt_evaluate_prompt)
                    if gt_score <= 0.5:
                        break
                    scores.append(gt_score)
                if len(scores) < len(profile_score_infos):
                    continue
                valid_rubrics.append(rubric)
            if len(valid_rubrics) < 2 or len(valid_rubrics) > 10:
                return {
                    "gt_score": 0.0,
                    "pd_non_score": 0.0,
                    "pd_rag_score": 0.0,
                    "pd_sft_score": 0.0,
                    "pd_grpo_score": 0.0,
                    "pd_sft_grpo_score": 0.0,
                    "total_rubrics": len(rubrics),
                    "valid_rubrics": len(valid_rubrics),
                }
            gt_scores = []
            pd_non_scores = []
            pd_rag_scores = []
            pd_sft_scores = []
            pd_grpo_scores = []
            pd_sft_grpo_scores = []
            for rubric in valid_rubrics:
                gt_prompt = build_rubrics_evaluate_prompt(data_source, metadata, gt, rubric)
                pd_non_prompt = build_rubrics_evaluate_prompt(data_source, metadata, pd_non, rubric)
                pd_rag_prompt = build_rubrics_evaluate_prompt(data_source, metadata, pd_rag, rubric)
                pd_sft_prompt = build_rubrics_evaluate_prompt(data_source, metadata, pd_sft, rubric)
                pd_grpo_prompt = build_rubrics_evaluate_prompt(data_source, metadata, pd_grpo, rubric)
                pd_sft_grpo_prompt = build_rubrics_evaluate_prompt(data_source, metadata, pd_sft_grpo, rubric)
                gt_score = await evaluate_score(client, gt_prompt)
                pd_non_score = await evaluate_score(client, pd_non_prompt)
                pd_rag_score = await evaluate_score(client, pd_rag_prompt)
                pd_sft_score = await evaluate_score(client, pd_sft_prompt)
                pd_grpo_score = await evaluate_score(client, pd_grpo_prompt)
                pd_sft_grpo_score = await evaluate_score(client, pd_sft_grpo_prompt)
                gt_scores.append(gt_score)
                pd_non_scores.append(pd_non_score)
                pd_rag_scores.append(pd_rag_score)
                pd_sft_scores.append(pd_sft_score)
                pd_grpo_scores.append(pd_grpo_score)
                pd_sft_grpo_scores.append(pd_sft_grpo_score)
            return {
                "gt_score": float(np.mean(gt_scores)),
                "pd_non_score": float(np.mean(pd_non_scores)),
                "pd_rag_score": float(np.mean(pd_rag_scores)),
                "pd_sft_score": float(np.mean(pd_sft_scores)),
                "pd_grpo_score": float(np.mean(pd_grpo_scores)),
                "pd_sft_grpo_score": float(np.mean(pd_sft_grpo_scores)),
                "total_rubrics": len(rubrics),
                "valid_rubrics": len(valid_rubrics),
            }

def _run_async_rubrics(args):
    with GLOBAL_SEM:
        return asyncio.run(process_sample_rubrics(args))

def get_reward(results):
    rewards = []
    for result in results:
        valid_rubrics = result["valid_rubrics"]
        total_rubrics = result["total_rubrics"]
        if total_rubrics < 2 or total_rubrics > 10 or valid_rubrics < 2 or valid_rubrics > 10:
            data = {
                "score": 0.1 if total_rubrics > 0 else 0.0,
                "r_diff": 0.0,
                "gt_score": 0.0,
                "pd_non_score": 0.0,
                "pd_rag_score": 0.0,
                "pd_sft_score": 0.0,
                "pd_grpo_score": 0.0,
                "pd_sft_grpo_score": 0.0,
                "cardinality": 0,
                "total_rubrics": total_rubrics,
                "valid_rubrics": valid_rubrics,
            }
            rewards.append(data)
            continue
        k = 20
        cardinality = 1 if valid_rubrics >= 2 else valid_rubrics / 2
        gt_score = result["gt_score"]
        pd_non_score = result["pd_non_score"]
        pd_rag_score = result["pd_rag_score"]
        pd_sft_score = result["pd_sft_score"]
        pd_grpo_score = result["pd_grpo_score"]
        pd_sft_grpo_score = result["pd_sft_grpo_score"]
        baselines = torch.tensor([pd_non_score, pd_rag_score, pd_sft_score, pd_grpo_score, pd_sft_grpo_score])
        advantage = gt_score - baselines
        log_probs = torch.sigmoid(k * advantage)
        r_diff = torch.prod(log_probs).item()
        score = 0.1 + 10 * gt_score * cardinality * r_diff
        data = {
            "score": score,
            "r_diff": r_diff,
            "gt_score": gt_score,
            "pd_non_score": pd_non_score,
            "pd_rag_score": pd_rag_score,
            "pd_sft_score": pd_sft_score,
            "pd_grpo_score": pd_grpo_score,
            "pd_sft_grpo_score": pd_sft_grpo_score,
            "cardinality": cardinality,
            "total_rubrics": total_rubrics,
            "valid_rubrics": valid_rubrics,
        }
        rewards.append(data)
    return rewards

def compute_rubrics_score(data_sources, solution_strs, ground_truths, extra_infos):
    preds, refs = solution_strs, ground_truths
    preds = [postprocess(pred) for pred in preds]
    num_samples = len(data_sources)

    thread_pool = ThreadPoolExecutor(max_workers=NUM_WORKERS)
    
    rubrics_futures = {}
    for idx in range(num_samples):
        gt = ground_truths[idx]
        metadata = extra_infos[idx]["metadata"]
        pd_non = extra_infos[idx]["baseline"]["pd_non"]
        pd_rag = extra_infos[idx]["baseline"]["pd_rag"]
        pd_sft = extra_infos[idx]["baseline"]["pd_sft"]
        pd_grpo = extra_infos[idx]["baseline"]["pd_grpo"]
        pd_sft_grpo = extra_infos[idx]["baseline"]["pd_sft_grpo"]
        profile = extra_infos[idx]['profile']
        num = min(len(profile), 10)
        profile = profile[-num:]
        response = preds[idx]
        response = postprocess(response)
        gens = (gt, pd_non, pd_rag, pd_sft, pd_grpo, pd_sft_grpo)
        task_arg = (data_sources[idx], metadata, response, gens, profile)
        rubrics_futures[idx] = thread_pool.submit(_run_async_rubrics, task_arg)
    
    results = []
    for idx, future in rubrics_futures.items():
        result = future.result()
        results.append(result)
    rewards = get_reward(results)
    return rewards

import os
import json
import jsonlines
import re
from pathlib import Path
def normalize_text(text):
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text).strip()
def extract_prompts_for_task(eval_file_path, task_name, mid_result_filename):
    questions_to_extract = []
    eval_data_list = []
    normalized_questions_map = {}
    with jsonlines.open(eval_file_path) as reader:
        for item in reader:
            q = item.get('question') or item.get('query')
            if not q:
                continue
            questions_to_extract.append(q)
            eval_data_list.append(item)
            normalized_questions_map[normalize_text(q)] = q
    print(f"✅ [{task_name}] 需要提取的 {len(questions_to_extract)} 个问题已加载。")
    methods = ["sparsecl", "bsim", "sentli", "her"]
    base_dir = "/home/vincent/hyper-simulation/data/mid_result"
    output_dir = f"/home/vincent/hyper-simulation/data/lagel_multihop/{task_name}"
    os.makedirs(os.path.join(output_dir, "prompts"), exist_ok=True)
    not_extracted_norm_q_global = {normalize_text(q) for q in questions_to_extract}
    for method in methods:
        file_path = os.path.join(base_dir, method, mid_result_filename)
        output_prompts = os.path.join(output_dir, "prompts", f"{method}_prompts.jsonl")
        if not os.path.exists(file_path):
            print(f"  ⚠️ {file_path} 不存在，跳过。")
            continue
        extracted_data = []
        with jsonlines.open(file_path) as reader:
            for item in reader:
                q = item.get("question") or item.get("query")
                norm_q = normalize_text(q)
                if norm_q in normalized_questions_map:
                    item["question"] = normalized_questions_map[norm_q]
                    extracted_data.append(item)
        extracted_norm_q = {normalize_text(item.get("question") or item.get("query")) for item in extracted_data}
        all_norm_q = {normalize_text(q) for q in questions_to_extract}
        not_extracted_norm_q_global = not_extracted_norm_q_global.intersection(all_norm_q - extracted_norm_q)
        prompt_by_norm_q = {normalize_text(item.get("question") or item.get("query")): item for item in extracted_data}
        final_prompts = []
        for eval_item in eval_data_list:
            q = eval_item.get("question") or eval_item.get("query")
            norm_q = normalize_text(q)
            if norm_q in prompt_by_norm_q:
                final_prompts.append(prompt_by_norm_q[norm_q].copy())
        with jsonlines.open(output_prompts, 'w') as writer:
            writer.write_all(final_prompts)
        print(f"  🎯 成功为方法 {method} 提取 {len(final_prompts)} 个 prompt")
    output_data = os.path.join(output_dir, "data.jsonl")
    not_extracted_data = []
    for item in eval_data_list:
        q = item.get("question") or item.get("query")
        if normalize_text(q) in not_extracted_norm_q_global:
            not_extracted_data.append(item)
    with jsonlines.open(output_data, 'w') as writer:
        writer.write_all(not_extracted_data)
    print(f"  🎯 成功将 {len(not_extracted_data)} 个未匹配的原始数据保存到 {output_data}")
    print("-" * 50)
def main():
    tasks = [
        {
            "name": "hotpot_distractor_easy",
            "eval_path": "/home/vincent/hyper-simulation/data/hotpotqa/hotpot_distractor_easy.jsonl",
            "mid_file": "hotpotqa.jsonl"
        },
        {
            "name": "hotpot_distractor_medium",
            "eval_path": "/home/vincent/hyper-simulation/data/hotpotqa/hotpot_distractor_medium.jsonl",
            "mid_file": "hotpotqa.jsonl"
        },
        {
            "name": "hotpot_distractor_hard",
            "eval_path": "/home/vincent/hyper-simulation/data/hotpotqa/hotpot_distractor_hard.jsonl",
            "mid_file": "hotpotqa.jsonl"
        }
    ]
    for task in tasks:
        extract_prompts_for_task(
            eval_file_path=task["eval_path"],
            task_name=task["name"],
            mid_result_filename=task["mid_file"]
        )
if __name__ == "__main__":
    main()
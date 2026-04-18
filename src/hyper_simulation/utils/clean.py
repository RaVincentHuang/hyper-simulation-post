import re
import os
import json
from pathlib import Path
def clean_text_for_spacy(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\[n\s+\d+\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(r'[–—―]', '-', text)
    return text
def deduplicate_jsonl_files():
    target_dirs = [
        "/home/vincent/hyper-simulation/data/baseline",
        "/home/vincent/hyper-simulation/data/mid_result",
        "/home/vincent/.dataset/LegalBench/sample500",
        "/home/vincent/.dataset/ARC/sample_ARC",
        "/home/vincent/.dataset/MultiHop/sample2500",
        "/home/vincent/.dataset/musique/sample3000",
        "/home/vincent/.dataset/HotpotQA/sample1000"
    ]
    for target_dir in target_dirs:
        if not os.path.exists(target_dir):
            print(f"⚠️ 目录不存在跳过: {target_dir}")
            continue
        for filepath in Path(target_dir).rglob("*.jsonl"):
            print(f"🔄 正在处理: {filepath}")
            seen_questions = set()
            unique_lines = []
            original_count = 0
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        original_count += 1
                        try:
                            data = json.loads(line)
                            question = data.get('question', '')
                            if not question or question not in seen_questions:
                                if question:
                                    seen_questions.add(question)
                                unique_lines.append(line)
                        except json.JSONDecodeError:
                            unique_lines.append(line)
                if len(unique_lines) < original_count:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        for line in unique_lines:
                            f.write(line + '\n')
                    print(f"  ✅ 完成去重: 原有 {original_count} 行 -> 去重后 {len(unique_lines)} 行 (删除了 {original_count - len(unique_lines)} 条重复数据)")
                else:
                    print(f"  ✨ 无需去重: 文件共 {original_count} 行，没有发现重复的 question。")
            except Exception as e:
                print(f"  ❌ 处理文件 {filepath} 时出错: {e}")
def deduplicate_json_files():
    target_dirs = [
        "/home/vincent/hyper-simulation/data/baseline",
        "/home/vincent/hyper-simulation/data/mid_result",
        "/home/vincent/.dataset/LegalBench/sample500",
        "/home/vincent/.dataset/ARC/sample_ARC",
        "/home/vincent/.dataset/MultiHop/sample2500",
        "/home/vincent/.dataset/musique/sample3000",
        "/home/vincent/.dataset/HotpotQA/sample1000"
    ]
    for target_dir in target_dirs:
        if not os.path.exists(target_dir):
            continue
        for filepath in Path(target_dir).rglob("*.json"):
            print(f"🔄 正在处理 JSON: {filepath}")
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if "results" not in data or not isinstance(data["results"], list):
                    print(f"  ⏭️ 跳过: 格式不符合预期 (缺少 results 列表)")
                    continue
                results = data["results"]
                original_count = len(results)
                seen_questions = set()
                unique_results = []
                for item in results:
                    question = item.get('question', '')
                    if not question or question not in seen_questions:
                        if question:
                            seen_questions.add(question)
                        unique_results.append(item)
                if len(unique_results) < original_count or data.get("total_processed") != len(unique_results):
                    data["results"] = unique_results
                    data["total_processed"] = len(unique_results)
                    if "config" in data:
                        data["config"]["total_samples_original"] = len(unique_results)
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    print(f"  ✅ 完成去重并更新统计: 原有 {original_count} 条 -> 去重后 {len(unique_results)} 条 (已更新 total_processed 和 total_samples_original)")
                else:
                    needs_update = False
                    if data.get("total_processed") != original_count:
                        data["total_processed"] = original_count
                        needs_update = True
                    if "config" in data and data["config"].get("total_samples_original") != original_count:
                        data["config"]["total_samples_original"] = original_count
                        needs_update = True
                    if needs_update:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                        print(f"  🔧 修复了文件 {filepath} 中的统计字段，统一为 {original_count}")
                    else:
                        print(f"  ✨ 无需去重: 文件共 {original_count} 条，没有发现重复的 question。")
            except json.JSONDecodeError:
                print(f"  ❌ 跳过: 无法解析 JSON 格式文件 {filepath}")
            except Exception as e:
                print(f"  ❌ 处理文件 {filepath} 时出错: {e}")
def fix_arc_baseline_metrics():
    import jsonlines
    from pathlib import Path
    from hyper_simulation.question_answer.utils.post_answer import evaluate_answer
    answer_map = {}
    reference_file = "/home/vincent/hyper-simulation/data/retr_result/arc/arc_with_context.jsonl"
    if os.path.exists(reference_file):
        with jsonlines.open(reference_file) as reader:
            for item in reader:
                q = item.get('question', '').strip()
                ans = item.get('answerKey', '')
                if isinstance(ans, list) and ans:
                    ans = ans[0]
                if q and ans:
                    answer_map[q] = ans
    orig_file = "/home/vincent/hyper-simulation/data/eval_data/arc_challenge_processed.jsonl"
    if os.path.exists(orig_file):
        with jsonlines.open(orig_file) as reader:
            for item in reader:
                q = item.get('question', '').strip()
                ans = item.get('answerKey', '')
                if isinstance(ans, list) and ans:
                    ans = ans[0]
                if q and ans:
                    answer_map[q] = ans
    dataset_dir = Path("/home/vincent/.dataset/ARC/sample_ARC")
    if dataset_dir.exists():
        for file_path in dataset_dir.glob("*.jsonl"):
            with jsonlines.open(file_path) as reader:
                for item in reader:
                    q = item.get('question', '').strip()
                    ans = item.get('answerKey', '')
                    if isinstance(ans, list) and ans:
                        ans = ans[0]
                    if q and ans:
                        answer_map[q] = ans
    baseline_dir = Path("/home/vincent/hyper-simulation/data/baseline")
    for arc_file in baseline_dir.rglob("ARC.json"):
        print(f"🔧 正在修复并清理 {arc_file}...")
        try:
            with open(arc_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if "results" not in data:
                continue
            seen_questions = set()
            fixed_results = []
            for item in data["results"]:
                q_raw = item.get("question", "")
                q_clean = q_raw.split('\n\nOptions:')[0].strip()
                if not q_clean or q_clean in seen_questions:
                    continue
                seen_questions.add(q_clean)
                current_ans = item.get("reference_answer", [])
                if not current_ans or current_ans == ["[]"] or current_ans == []:
                    ans = answer_map.get(q_clean)
                    if not ans:
                        for mq, ma in answer_map.items():
                            if q_clean.startswith(mq) or mq.startswith(q_clean):
                                ans = ma
                                break
                    if ans:
                        item["reference_answer"] = [ans]
                    else:
                        print(f"⚠️ 找不到问题答案: {q_clean[:50]}...")
                pred = item.get("prediction", "")
                ref = item.get("reference_answer", [])
                if ref == ["[]"]:
                    ref = []
                if ref:
                    metrics = evaluate_answer(pred, ref)
                    item["metrics"] = metrics
                    item["is_correct"] = metrics["exact_match"] > 0
                else:
                    item["metrics"] = {"exact_match": False, "f1": 0, "match": 0}
                    item["is_correct"] = False
                fixed_results.append(item)
            data["results"] = fixed_results
            data["total_processed"] = len(fixed_results)
            if "config" in data:
                data["config"]["total_samples_original"] = len(fixed_results)
            all_metrics = {"exact_match": [], "f1": [], "match": []}
            for item in fixed_results:
                m = item.get("metrics", {})
                for k in all_metrics:
                    if k in m:
                        all_metrics[k].append(m[k])
            data["avg_metrics"] = {
                k: sum(v)/len(v) if v else 0 
                for k, v in all_metrics.items()
            }
            with open(arc_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"  ✅ 修复完成，共计 {len(fixed_results)} 条。Accuracy: {data['avg_metrics'].get('exact_match', 0):.4f}")
        except Exception as e:
            print(f"❌ 修复文件 {arc_file} 失败: {e}")
def deduplicate_musique_eval_data():
    filepath = "/home/vincent/hyper-simulation/data/eval_data/musique_answerable.jsonl"
    if not os.path.exists(filepath):
        print(f"⚠️ {filepath} 不存在，跳过。")
        return
    print(f"🔄 正在处理 musique 评测数据: {filepath}")
    seen_ids = set()
    unique_lines = []
    original_count = 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                original_count += 1
                try:
                    data = json.loads(line)
                    item_id = data.get('question')
                    if item_id is None:
                        unique_lines.append(line)
                    elif item_id not in seen_ids:
                        seen_ids.add(item_id)
                        unique_lines.append(line)
                except json.JSONDecodeError:
                    unique_lines.append(line)
        if len(unique_lines) < original_count:
            with open(filepath, 'w', encoding='utf-8') as f:
                for line in unique_lines:
                    f.write(line + '\n')
            print(f"  ✅ 完成去重: 原有 {original_count} 行 -> 去重后 {len(unique_lines)} 行 (删除了 {original_count - len(unique_lines)} 条重复数据)")
        else:
            print(f"  ✨ 无需去重: 文件共 {original_count} 行，没有发现重复的 id。")
    except Exception as e:
        print(f"  ❌ 处理文件时出错: {e}")
if __name__ == "__main__":
    print("开始执行 musique_answerable.jsonl 的专属去重任务...")
    deduplicate_musique_eval_data()
    print("\n去重任务执行完毕。")
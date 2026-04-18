import torch
import argparse
import os
import json
import jsonlines
from types import SimpleNamespace
from typing import List, Dict, Union
import glob
import time
from pathlib import Path
from hyper_simulation.question_answer.vmdit.retrieval import (
    embed_queries, 
    add_passages, 
    add_hasanswer,
    index_encoded_data
)
import contrievers
import contrievers.index
import contrievers.data
from hyper_simulation.llm.chat_completion import get_generate
from langchain_ollama import ChatOllama
from hyper_simulation.question_answer.vmdit.utils import (
    PROMPT_DICT, 
    TASK_INST, 
    postprocess_answers_closed,
    preprocess_input_data
)
class RAGPipeline:
    def __init__(self, 
                 retriever_model_path: str = "models/contriever-msmarco",
                 passages_path: str = "data/psgs_w100.tsv",
                 index_path: str = "index_hnsw/",
                 embedding_dir: str = "data/wikipedia_embeddings",
                 llm_model_name: str = "qwen3.5:9b",
                 device: str = "cuda"):
        self.device = device
        print(f"Loading Retriever from {retriever_model_path}...")
        self.retriever_model, self.retriever_tokenizer, _ = contrievers.load_retriever(retriever_model_path)
        self.retriever_model.eval()
        self.retriever_model.to(device)
        if device == "cuda":
            self.retriever_model.half()
        print(f"Loading Index from {index_path}...")
        self.index = contrievers.index.Indexer(vector_sz=768, n_subquantizers=0, n_bits=8, mode='hnsw')
        index_dir = index_path.rstrip('/')
        index_file = os.path.join(index_dir, "index.faiss")
        meta_file = os.path.join(index_dir, "index_meta.faiss")
        if os.path.exists(index_file):
            import faiss
            import pickle
            print(f"⚡ Loading 65GB index via Memory-Mapped I/O (MMAP) to bypass RAM limit...")
            faiss_idx = faiss.read_index(index_file, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
            target_attr = "index" if hasattr(self.index, "index") else "faiss_index"
            setattr(self.index, target_attr, faiss_idx)
            if os.path.exists(meta_file):
                print(f"Loading meta data from {meta_file}")
                with open(meta_file, "rb") as reader:
                    self.index.index_id_to_db_id = pickle.load(reader)
            else:
                print(f"⚠️ Warning: Meta data not found at {meta_file}")
            print(f"✅ Index mapped successfully. Physical RAM usage stable.")            
        else:
            print(f"Index not found at {index_path}. Building from embeddings in {embedding_dir}...")
            input_paths = glob.glob(os.path.join(embedding_dir, "passages_*")) 
            input_paths = sorted(input_paths)
            if not input_paths:
                 raise FileNotFoundError(f"No embedding files found in {embedding_dir}. Please run generate_passage_embedding.py first.")
            start_time = time.time()
            index_encoded_data(self.index, input_paths, indexing_batch_size=1000000)
            print(f"Indexing finished in {time.time()-start_time:.1f} s.")
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            self.index.serialize(index_path)
            print(f"Index saved to {index_path}")
        print(f"Loading Passages from {passages_path}...")
        self.passages = contrievers.data.load_passages(passages_path)
        self.passage_id_map = {x["id"]: x for x in self.passages}
        print(f"Loading LLM {llm_model_name}...")
        self.llm = ChatOllama(model=llm_model_name, temperature=0.8, top_p=0.95)
    def retrieve(self, queries: List[str], top_k: int = 5) -> List[List[Dict]]:
        args = SimpleNamespace(
            lowercase=False, 
            normalize_text=True, 
            per_gpu_batch_size=32, 
            question_maxlength=512
        )
        print("Embedding queries...")
        query_embeddings = embed_queries(args, queries, self.retriever_model, self.retriever_tokenizer)
        print("Searching index...")
        top_ids_and_scores = self.index.search_knn(query_embeddings, top_k)
        dummy_data = [{"question": q} for q in queries]
        add_passages(dummy_data, self.passage_id_map, top_ids_and_scores)
        return [item["ctxs"] for item in dummy_data]
    def generate(self, items: List[Dict], task: str = "qa", top_n: int = 5, save_prompts_only: bool = False, prompt_save_path: str = None) -> List[str]:
        prompts = []
        for item in items:
            retrieval_result = item.get("ctxs", [])[:top_n]
            evidences = [
                "[{}] {}\n{}".format(i+1, ctx["title"], ctx["text"]) 
                for i, ctx in enumerate(retrieval_result)
            ]
            paragraph = "\n".join(evidences)
            instruction_text = TASK_INST.get(task, item.get("question", ""))
            choices_str = ""
            if task in ["arc_c", "arc_easy", "obqa"] and "choices" in item:
                choices = item["choices"]
                labels = choices.get("label", [])
                texts = choices.get("text", [])
                formatted = []
                map_key = {"1": "A", "2": "B", "3": "C", "4": "D"}
                for l, t in zip(labels, texts):
                    k = map_key.get(l, l)
                    formatted.append(f"{k}: {t}")
                if formatted:
                    choices_str = "\n" + "\n".join(formatted)
            full_instruction = f"{instruction_text}\n\n### Input:\n{item['question']}{choices_str}"
            prompt = PROMPT_DICT["prompt_no_input_retrieval"].format(
                paragraph=paragraph,
                instruction=full_instruction
            )
            prompts.append(prompt)
        if save_prompts_only and prompt_save_path:
            prompts_buffer = []
            for item in items:
                retrieval_result = item.get("ctxs", [])[:top_n]
                paragraphs = []
                for ctx in retrieval_result:
                    title = ctx.get("title", "")
                    text = ctx.get("text", "")
                    paragraphs.append({
                        "title": title,
                        "text": text,
                        "is_supporting": True
                    })
                prompt_entry = {
                    "question": item.get("question", ""),
                    "answerKey": item.get("answers", []),
                    "choices": item.get("choices", {}),
                    "paragraphs": paragraphs
                }
                prompts_buffer.append(prompt_entry)
            Path(prompt_save_path).parent.mkdir(parents=True, exist_ok=True)
            with jsonlines.open(prompt_save_path, 'a') as writer:
                for entry in prompts_buffer:
                    writer.write(entry)
            print(f"💾 已保存 {len(prompts_buffer)} 条带 Retrieval Context 的数据到 {prompt_save_path}")
            return [""] * len(items)
        print(f"Generating responses for {len(prompts)} prompts...")
        raw_responses = get_generate(prompts, self.llm)
        print(f"Raw responses is {raw_responses}")
        final_results = []
        for resp in raw_responses:
            cleaned = resp.split("\n\n")[0].replace("</s>", "").strip()
            choices_arg = "A B C D" if task in ["arc_c", "arc_easy"] else None
            final_out = postprocess_answers_closed(cleaned, task, choices=choices_arg)
            final_results.append(final_out)
        return final_results
    def run_batch(self, input_data: List[Dict], task: str = "qa", top_n: int = 5, save_prompts_only: bool = False, prompt_save_path: str = None):
        queries = [item["question"] for item in input_data]
        print("--- Start Retrieval ---")
        ctxs_list = self.retrieve(queries, top_k=top_n)
        for item, ctxs in zip(input_data, ctxs_list):
            item["ctxs"] = ctxs
        print("--- Start Generation ---")
        answers = self.generate(input_data, task=task, top_n=top_n, save_prompts_only=save_prompts_only, prompt_save_path=prompt_save_path)
        for item, ans in zip(input_data, answers):
            item["output"] = ans
        return input_data
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None, help="Path to ARC data")
    parser.add_argument('--save_prompts_only', action='store_true', help="Only retrieve and save prompts")
    parser.add_argument('--prompt_save_path', type=str, default="/home/vincent/hyper-simulation/data/mid_result/arc/arc_retrieved.jsonl")
    args = parser.parse_args()
    rag = RAGPipeline(
        retriever_model_path="models/contriever-msmarco",
        passages_path="data/psgs_w100.tsv",
        index_path="../index_hnsw/"
    )
    if args.data_path:
        from hyper_simulation.question_answer.utils.load_data import load_data
        raw_data = load_data(args.data_path, "ARC")
        print(f"Loaded {len(raw_data)} samples from {args.data_path}")
        test_data = []
        for item in raw_data:
            test_data.append({
                "question": item.get("question", ""),
                "choices": item.get("choices", {}),
                "answers": item.get("answerKey", [])
            })
        batch_size = 100
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i+batch_size]
            print(f"Processing batch {i} to {i+len(batch)}...")
            rag.run_batch(
                batch, 
                task="arc_c", 
                top_n=5, 
                save_prompts_only=args.save_prompts_only,
                prompt_save_path=args.prompt_save_path
            )
        print("Done!")
    else:
        test_data = [
            {
                "id": 1,
                "question": "what is the capital of China?",
            },
            {
                "id": 2,
                "question": "Which material conducts heat best?",
                "choices": {"text": ["Wood", "Copper", "Plastic", "Glass"], "label": ["A", "B", "C", "D"]}
            }
        ]
        results = rag.run_batch(test_data[:1], task="qa")
        print(f"QA Result: {results[0]['output']}")
        results_arc = rag.run_batch(test_data[1:], task="arc_c")
        print(f"ARC Result: {results_arc[0]['output']}")
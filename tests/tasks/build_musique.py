import time
from argparse import ArgumentParser
import json
from pathlib import Path
from turtle import st
from typing import Iterator
import logging
import spacy
from tqdm import tqdm
from tqdm.auto import tqdm as tqdm_auto
from spacy.language import Language
from hyper_simulation.component.build_hypergraph import (
    generate_instance_id, 
    text_to_hypergraph,
    doc_to_hypergraph,
    clean_text_for_spacy,
)
from hyper_simulation.question_answer.utils.load_data import load_data
from hyper_simulation.query_instance import build_query_instance_for_task
logger = logging.getLogger(__name__)
target_dir = "data/debug/musique/sample1417/"
dataset_path = "/home/vincent/.dataset/musique/rest/musique_answerable.jsonl"
local_model_path = "/home/vincent/.cache/huggingface/hub/models--biu-nlp--lingmess-coref/snapshots/fa5d8a827a09388d03adbe9e800c7d8c509c3935"
def setup_gpu_nlp(model_name: str = "en_core_web_trf") -> Language:
	try:
		spacy.require_gpu()
		logger.info("✅ GPU enabled for spaCy")
	except Exception as e:
		logger.warning(f"⚠️ GPU initialization failed: {e}, falling back to CPU")
	try:
		nlp = spacy.load(model_name)
		logger.info(f"✅ Loaded spaCy model: {model_name}")
	except OSError:
		logger.error(f"❌ Model {model_name} not found. Please run: python -m spacy download {model_name}")
		raise
	if "fastcoref" not in nlp.pipe_names:
		try:
			nlp.add_pipe(
				"fastcoref",
				config={
					"model_architecture": "LingMessCoref",
					"model_path": local_model_path,
					"device": "cuda",
				}
			)
			logger.info("✅ Added fastcoref with CUDA support")
		except Exception as e:
			logger.warning(f"⚠️ Failed to add fastcoref: {e}")
	return nlp
def batch_text_to_hypergraph(
	nlp: Language,
	texts_with_metadata: list[dict],
	batch_size: int = 32,
	is_query: bool = False,
) -> Iterator[tuple[dict, object]]:
	texts = [clean_text_for_spacy(item["text"]) for item in texts_with_metadata]
	metadatas = [item["meta"] for item in texts_with_metadata]
	original_texts = [item["text"] for item in texts_with_metadata]
	component_cfg = {"fastcoref": {"resolve_text": True}} if "fastcoref" in nlp.pipe_names else {}
	try:
		docs_list = list(
			nlp.pipe(
				texts,
				component_cfg=component_cfg,
				batch_size=max(1, batch_size),
			)
		)
		for doc, metadata, original_text in zip(docs_list, metadatas, original_texts):
			try:
				hypergraph = doc_to_hypergraph(doc, original_text, is_query=is_query)
				yield metadata, hypergraph
			except Exception as e:
				error_msg = f"{type(e).__name__}: {e}"
				metadata["error"] = error_msg
				logger.error(f"Error converting doc to hypergraph: {error_msg}")
				yield metadata, None
	except Exception as e:
		logger.warning(
			f"⚠️ Batch processing failed: {type(e).__name__}: {e}. "
			"Falling back to per-text processing."
		)
		for text, metadata, original_text in zip(texts, metadatas, original_texts):
			try:
				doc = nlp(text)
				hypergraph = doc_to_hypergraph(doc, original_text, is_query=is_query)
				yield metadata, hypergraph
			except Exception as e2:
				error_msg = f"{type(e2).__name__}: {e2}"
				metadata["error"] = error_msg
				logger.error(f"Error processing individual text: {error_msg}")
				yield metadata, None
def _build_all_hypergraphs_single(
	dataset_path: str,
	target_dir: str,
	force_rebuild: bool = False,
	using_support_only: bool = False,
	save_outputs: bool = True,
) -> dict:
	data = load_data(dataset_path, task="musique", use_supporting_only=using_support_only)
	out_root = Path(target_dir)
	if save_outputs:
		out_root.mkdir(parents=True, exist_ok=True)
	built_count = 0
	skipped_count = 0
	failed: list[dict] = []
	for item in tqdm(data, desc="Building musique hypergraphs (single)"):
		try:
			qi = build_query_instance_for_task(item, task="musique")
			question = (qi.query or "").strip()
			if not question:
				skipped_count += 1
				continue
			instance_id = generate_instance_id(question)
			instance_dir = out_root / instance_id
			query_path = instance_dir / "query_hypergraph.pkl"
			metadata_path = instance_dir / "metadata.json"
			if save_outputs and metadata_path.exists() and not force_rebuild:
				skipped_count += 1
				continue
			query_hypergraph = text_to_hypergraph(question, is_query=True)
			if save_outputs:
				instance_dir.mkdir(parents=True, exist_ok=True)
				query_hypergraph.save(str(query_path))
			data_files = []
			for idx, doc_text in enumerate(qi.data):
				text = (doc_text or "").strip()
				if not text:
					continue
				data_hypergraph = text_to_hypergraph(text, is_query=False)
				data_file = f"data_hypergraph{idx}.pkl"
				if save_outputs:
					data_hypergraph.save(str(instance_dir / data_file))
				data_files.append(data_file)
			metadata = {
				"instance_id": instance_id,
				"source_id": item.get("_id", ""),
				"question": question,
				"num_data": len(qi.data),
				"saved_data_hypergraphs": len(data_files),
				"files": {
					"query": query_path.name,
					"data": data_files,
				},
			}
			if save_outputs:
				metadata_path.write_text(
					json.dumps(metadata, indent=2, ensure_ascii=False),
					encoding="utf-8",
				)
			built_count += 1
		except Exception as exc:
			failed.append(
				{
					"id": item.get("_id", ""),
					"question": item.get("question", ""),
					"error": f"{type(exc).__name__}: {exc}",
				}
			)
	summary = {
		"dataset_path": dataset_path,
		"target_dir": str(out_root.resolve()),
		"total_questions": len(data),
		"built": built_count,
		"skipped": skipped_count,
		"failed": len(failed),
		"mode": "single",
		"save_outputs": save_outputs,
	}
	if save_outputs:
		(out_root / "summary.json").write_text(
			json.dumps(summary, indent=2, ensure_ascii=False),
			encoding="utf-8",
		)
		if failed:
			(out_root / "failed.json").write_text(
				json.dumps(failed, indent=2, ensure_ascii=False),
				encoding="utf-8",
			)
	return summary
def _build_all_hypergraphs_gpu_batch(
	dataset_path: str,
	target_dir: str,
	force_rebuild: bool = False,
	using_support_only: bool = False,
	batch_size: int = 32,
	save_outputs: bool = True,
) -> dict:
	logger.info("🚀 Starting GPU-accelerated batch processing...")
	time_cost= 0.0
	nlp = setup_gpu_nlp()
	data = load_data(dataset_path, task="musique", use_supporting_only=using_support_only)
	out_root = Path(target_dir)
	if save_outputs:
		out_root.mkdir(parents=True, exist_ok=True)
	built_count = 0
	skipped_count = 0
	failed: list[dict] = []
	logger.info("📋 [阶段1/4] 收集文本...")
	queries_to_process = []
	contexts_to_process = []
	for item_idx, item in enumerate(tqdm(data, desc="[阶段1/4] 扫描数据", unit="items")):
		try:
			qi = build_query_instance_for_task(item, task="musique")
			question = (qi.query or "").strip()
			if not question:
				skipped_count += 1
				continue
			instance_id = generate_instance_id(question)
			instance_dir = out_root / instance_id
			query_path = instance_dir / "query_hypergraph.pkl"
			metadata_path = instance_dir / "metadata.json"
			if save_outputs and metadata_path.exists() and not force_rebuild:
				skipped_count += 1
				continue
			queries_to_process.append((
				item,
				instance_id,
				instance_dir,
				query_path,
				metadata_path,
				question,
				qi,
			))
			for idx, doc_text in enumerate(qi.data):
				text = (doc_text or "").strip()
				if text:
					contexts_to_process.append((
						item,
						instance_id,
						instance_dir,
						text,
						idx,
						item_idx,
					))
		except Exception as exc:
			failed.append(
				{
					"id": item.get("_id", ""),
					"question": item.get("question", ""),
					"error": f"{type(exc).__name__}: {exc}",
				}
			)
	query_results = {}
	if queries_to_process:
		logger.info(f"⚙️ [阶段2/4] 处理查询 ({len(queries_to_process)} 个)...")
		queries_batch = [
			{
				"text": q[5],
				"meta": {
					"item": q[0],
					"instance_id": q[1],
					"query_path": q[3],
					"metadata_path": q[4],
					"qi": q[6],
				}
			}
			for q in queries_to_process
		]
		query_pbar = tqdm(desc="[阶段2/4] 转换查询超图", total=len(queries_batch), unit="queries")
		start_time = time.time()
		for metadata, hypergraph in batch_text_to_hypergraph(
			nlp,
			queries_batch,
			batch_size=batch_size,
			is_query=True,
		):
			if hypergraph is not None:
				query_results[metadata["instance_id"]] = (metadata, hypergraph)
			else:
				logger.error(f"Failed to process query for instance {metadata['instance_id']}")
			query_pbar.update(1)
		query_pbar.close()
		end_time = time.time()
		time_cost += (end_time - start_time)
	context_results = {}
	if contexts_to_process:
		logger.info(f"⚙️ [阶段3/4] 处理上下文 ({len(contexts_to_process)} 个)...")
		contexts_batch = [
			{
				"text": c[3],
				"meta": {
					"item": c[0],
					"instance_id": c[1],
					"instance_dir": c[2],
					"idx": c[4],
					"item_idx": c[5],
				}
			}
			for c in contexts_to_process
		]
		context_pbar = tqdm(desc="[阶段3/4] 转换上下文超图", total=len(contexts_batch), unit="contexts")
		start_time = time.time()
		for metadata, hypergraph in batch_text_to_hypergraph(
			nlp,
			contexts_batch,
			batch_size=batch_size,
			is_query=False,
		):
			if hypergraph is not None:
				key = (metadata["instance_id"], metadata["idx"])
				context_results[key] = hypergraph
			else:
				logger.error(f"Failed to process context for instance {metadata['instance_id']}")
			context_pbar.update(1)
		context_pbar.close()
		end_time = time.time()
		time_cost += (end_time - start_time)
	logger.info(f"💾 [阶段4/4] 保存结果 ({len(queries_to_process)} 个)...")
	for item, instance_id, instance_dir, query_path, metadata_path, question, qi in tqdm(
		queries_to_process,
		desc="[阶段4/4] 保存超图",
		unit="instances"
	):
		try:
			if instance_id not in query_results:
				logger.warning(f"No query result for instance {instance_id}")
				failed.append({
					"id": item.get("_id", ""),
					"question": question,
					"error": "Query hypergraph processing failed",
				})
				continue
			_, query_hypergraph = query_results[instance_id]
			if save_outputs:
				instance_dir.mkdir(parents=True, exist_ok=True)
				query_hypergraph.save(str(query_path))
			data_files = []
			for idx in range(len(qi.data)):
				key = (instance_id, idx)
				if key in context_results:
					data_hypergraph = context_results[key]
					data_file = f"data_hypergraph{idx}.pkl"
					if save_outputs:
						data_hypergraph.save(str(instance_dir / data_file))
					data_files.append(data_file)
			metadata = {
				"instance_id": instance_id,
				"source_id": item.get("_id", ""),
				"question": question,
				"num_data": len(qi.data),
				"saved_data_hypergraphs": len(data_files),
				"files": {
					"query": query_path.name,
					"data": data_files,
				},
			}
			if save_outputs:
				metadata_path.write_text(
					json.dumps(metadata, indent=2, ensure_ascii=False),
					encoding="utf-8",
				)
			built_count += 1
		except Exception as exc:
			failed.append(
				{
					"id": item.get("_id", ""),
					"question": question,
					"error": f"{type(exc).__name__}: {exc}",
				}
			)
	summary = {
		"dataset_path": dataset_path,
		"target_dir": str(out_root.resolve()),
		"total_questions": len(data),
		"built": built_count,
		"skipped": skipped_count,
		"failed": len(failed),
		"mode": "gpu_batch",
		"batch_size": batch_size,
		"save_outputs": save_outputs,
	}
	if save_outputs:
		(out_root / "summary.json").write_text(
			json.dumps(summary, indent=2, ensure_ascii=False),
			encoding="utf-8",
		)
		if failed:
			(out_root / "failed.json").write_text(
				json.dumps(failed, indent=2, ensure_ascii=False),
				encoding="utf-8",
			)
	print(f"Total time cost for GPU batch processing: {time_cost:.4f} seconds")
	print(f"Average time cost per instance: {time_cost / built_count:.4f} seconds (built_count={built_count})")
	return summary
def build_all_hypergraphs(
	dataset_path: str,
	target_dir: str,
	force_rebuild: bool = False,
	using_support_only: bool = False,
	use_gpu_batch: bool = False,
	batch_size: int = 32,
	save_outputs: bool = True,
) -> dict:
	if use_gpu_batch:
		return _build_all_hypergraphs_gpu_batch(
			dataset_path=dataset_path,
			target_dir=target_dir,
			force_rebuild=force_rebuild,
			using_support_only=using_support_only,
			batch_size=batch_size,
			save_outputs=save_outputs,
		)
	else:
		return _build_all_hypergraphs_single(
			dataset_path=dataset_path,
			target_dir=target_dir,
			force_rebuild=force_rebuild,
			using_support_only=using_support_only,
			save_outputs=save_outputs,
		)
def _print_summary(summary: dict) -> None:
	print("\n" + "="*70)
	print("✅ 超图构建完成！")
	print("="*70)
	total = summary.get("total_questions", 0)
	built = summary.get("built", 0)
	skipped = summary.get("skipped", 0)
	failed = summary.get("failed", 0)
	mode = summary.get("mode", "unknown")
	batch_size = summary.get("batch_size", "N/A")
	print(f"\n📊 处理统计:")
	print(f"  ├─ 总数:     {total}")
	print(f"  ├─ 成功:     {built} ✅")
	print(f"  ├─ 跳过:     {skipped} ⏭️")
	print(f"  └─ 失败:     {failed} ❌")
	print(f"\n⚙️  处理模式:")
	if mode == "gpu_batch":
		print(f"  ├─ 方式:     GPU 批处理 🚀")
		print(f"  └─ batch大小: {batch_size}")
	else:
		print(f"  └─ 方式:     单文本处理 🐢")
	print(f"\n📁 输出目录:")
	print(f"  └─ {summary.get('target_dir', 'N/A')}")
	print(f"\n📝 日志文件:")
	if summary.get("save_outputs", True):
		print(f"  ├─ summary.json (已保存)")
		if failed > 0:
			print(f"  └─ failed.json (包含 {failed} 条失败记录)")
	else:
		print(f"  └─ 未保存任何文件（execute-only 模式）")
	success_rate = (built / total * 100) if total > 0 else 0
	print(f"\n🎯 成功率: {success_rate:.1f}% ({built}/{total})")
	print("="*70 + "\n")
def main() -> None:
	parser = ArgumentParser(description="Build and store all MuSiQue question hypergraphs.")
	parser.add_argument("--dataset-path", type=str, default=dataset_path)
	parser.add_argument("--target-dir", type=str, default=target_dir)
	parser.add_argument("--force-rebuild", action="store_true")
	parser.add_argument("--using-support-only", action="store_true")
	parser.add_argument(
		"--execute-only",
		action="store_true",
		help="Run the MuSiQue build pipeline without saving any hypergraphs or metadata to disk",
	)
	parser.add_argument(
		"--use-gpu-batch",
		action="store_true",
		help="Enable GPU + batch processing for faster processing (requires CUDA and fastcoref)"
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=4096,
		help="Batch size for GPU batch processing (default: 32, recommended for fastcoref stability)"
	)
	args = parser.parse_args()
	print("\n" + "="*70)
	print("🚀 MuSiQue 超图构建工具")
	print("="*70)
	print(f"📂 数据集: {args.dataset_path}")
	print(f"📁 输出:   {args.target_dir}")
	print(f"🔧 模式:   {'GPU 批处理 🚀' if args.use_gpu_batch else '单文本处理 🐢'}")
	if args.use_gpu_batch:
		print(f"📦 batch大小: {args.batch_size}")
	print("="*70 + "\n")
	summary = build_all_hypergraphs(
		dataset_path=args.dataset_path,
		target_dir=args.target_dir,
		force_rebuild=args.force_rebuild,
		using_support_only=args.using_support_only,
		use_gpu_batch=args.use_gpu_batch,
		batch_size=args.batch_size,
		save_outputs=not args.execute_only,
	)
	_print_summary(summary)
if __name__ == "__main__":
	main()
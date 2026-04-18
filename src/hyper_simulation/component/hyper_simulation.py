import time
from typing import Dict, List, Set, Tuple
from hyper_simulation.hypergraph.hypergraph import Hypergraph as LocalHypergraph, Vertex
from simulation import Hypergraph as SimHypergraph, Hyperedge as SimHyperedge, Node, Delta, DMatch
from hyper_simulation.component.semantic_cluster import calc_semantic_cluster_pairs, get_d_match
from hyper_simulation.component.d_match import calc_d_match, calc_d_match_batch
from hyper_simulation.hypergraph.linguistic import Pos
from hyper_simulation.component.denial import get_matched_vertices, compute_allowed_pairs, compute_allowed_pairs_batch, compute_allowed_pairs_batch_with_score, get_top_k_matched_vertices_by_scores
import warnings
from tqdm import tqdm
from hyper_simulation.utils.log import getLogger
import logging
def convert_local_to_sim(
    local_hg: LocalHypergraph,
) -> Tuple[SimHypergraph, Dict[int, str], Dict[int, Vertex], Dict[int, List[SimHyperedge]], Dict[Vertex, int]]:
    sim_hg = SimHypergraph()
    vertex_id_map: Dict[int, int] = {}
    node_text: Dict[int, str] = {}
    sim_id_to_vertex: Dict[int, Vertex] = {}
    node_to_edges: Dict[int, List[SimHyperedge]] = {}
    vertex_to_sim_id: Dict[Vertex, int] = {}
    for idx, vertex in enumerate(sorted(local_hg.vertices, key=lambda v: v.id)):
        sim_hg.add_node(vertex.text())
        vertex_id_map[vertex.id] = idx
        node_text[idx] = vertex.text()
        sim_id_to_vertex[idx] = vertex
        vertex_to_sim_id[vertex] = idx
    edge_id = 0
    for local_edge in local_hg.hyperedges:
        node_ids = {vertex_id_map[v.id] for v in local_edge.vertices if v.id in vertex_id_map}
        if not node_ids:
            continue
        sim_edge = SimHyperedge(node_ids, local_edge.desc, edge_id)
        sim_hg.add_hyperedge(sim_edge)
        for nid in node_ids:
            node_to_edges.setdefault(nid, []).append(sim_edge)
        edge_id += 1
    return sim_hg, node_text, sim_id_to_vertex, node_to_edges, vertex_to_sim_id
def build_delta_and_dmatch(
    query: SimHypergraph,
    data: SimHypergraph,
    query_texts: Dict[int, str],
    data_texts: Dict[int, str],
    query_node_edges: Dict[int, List[SimHyperedge]],
    data_node_edges: Dict[int, List[SimHyperedge]],
    allowed_pairs: Set[Tuple[int, int]],
    query_local_hg: LocalHypergraph,
    data_local_hg: LocalHypergraph,
    vertex_to_sim_id_q: Dict[Vertex, int],
    vertex_to_sim_id_d: Dict[Vertex, int],
    matched_vertices: dict[Vertex, set[Tuple[Vertex, float]]],
    cluster_sim_threshold: float = 0.75,
    dmatch_threshold: float = 0.3,
    branch_threshold: int = 5,
    is_multihop: bool = False,
) -> Tuple[Delta, DMatch]:
    delta_start = time.time()
    sc_logger = getLogger("semantic_cluster") 
    sc_logger.debug(f"\t\tcalc the delta")
    delta = Delta()
    d_delta_matches: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}
    cluster_count = 0
    raw_pairs = calc_semantic_cluster_pairs(
        query_local_hg, data_local_hg, matched_vertices, 
        cluster_sim_threshold, branch_threshold, is_multihop, logger=sc_logger
        )
    time1 = time.time()
    sc_logger.info(f"语义簇生成完成: 共 {len(raw_pairs)} 个原始簇对")
    candidate_cluster_pairs = []
    for sc_q, sc_d, sim_score in raw_pairs:
        q_vertices = sc_q.get_vertices()
        d_vertices = sc_d.get_vertices()
        q_edges = sc_q.hyperedges
        d_edges = sc_d.hyperedges
        q_triples = sc_q.to_triple() or []
        d_triples = sc_d.to_triple() or []
        q_triple_repr = str(q_triples[0]) if q_triples else "(no triple)"
        d_triple_repr = str(d_triples[0]) if d_triples else "(no triple)"
        q_text = sc_q.text()
        d_text = sc_d.text()
        sc_logger.info(
            f"→ 原始簇对 | score={sim_score:.3f}\n"
            f"  Q: text='{q_text}'\n"
            f"     triple={q_triple_repr}\n"
            f"     nodes={len(q_vertices)}, edges={len(q_edges)}\n"
            f"  D: text='{d_text}'\n"
            f"     triple={d_triple_repr}\n"
            f"     nodes={len(d_vertices)}, edges={len(d_edges)}"
        )
        if sim_score < 0.5:
            sc_logger.info(f"  → 跳过: 低相似度 ({sim_score:.3f})")
            continue
        q_vs = [v for v in q_vertices if not (v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX))]
        d_vs = [v for v in d_vertices if not (v.pos_equal(Pos.VERB) or v.pos_equal(Pos.AUX))]
        if not q_vs or not d_vs:
            sc_logger.info(f"  → 跳过: 无名词节点 (Q:{len(q_vs)}/{len(q_vertices)}, D:{len(d_vs)}/{len(d_vertices)})")
            continue
        q_rep = min(q_vs, key=lambda v: v.id)
        d_rep = min(d_vs, key=lambda v: v.id)
        q_nid = vertex_to_sim_id_q.get(q_rep)
        d_nid = vertex_to_sim_id_d.get(d_rep)
        if q_nid is None or d_nid is None:
            sc_logger.info(f"  → 跳过: 映射缺失 (Q{q_rep.id}→{q_nid}, D{d_rep.id}→{d_nid})")
            continue
        q_es = list({e for v in q_vs if v in vertex_to_sim_id_q for e in query_node_edges.get(vertex_to_sim_id_q[v], []) if e})
        d_es = list({e for v in d_vs if v in vertex_to_sim_id_d for e in data_node_edges.get(vertex_to_sim_id_d[v], []) if e})
        sc_id = delta.add_sematic_cluster_pair(
            Node(q_nid, q_text),
            Node(d_nid, d_text),
            q_es,
            d_es
        )
        candidate_cluster_pairs.append({
            'sc_q': sc_q,
            'sc_d': sc_d,
            'sc_id': sc_id,
            'q_rep': q_rep,
            'd_rep': d_rep,
            'q_text': q_text,
            'd_text': d_text,
            'q_triple_repr': q_triple_repr,
            'd_triple_repr': d_triple_repr,
            'q_vertices': q_vertices,
            'd_vertices': d_vertices,
            'q_vs': q_vs,
            'd_vs': d_vs,
            'q_edges': q_edges,
            'd_edges': d_edges,
            'sim_score': sim_score,
        })
    if candidate_cluster_pairs:
        sc_pairs = [(md['sc_q'], md['sc_d']) for md in candidate_cluster_pairs]
        try:
            batch_results = calc_d_match_batch(sc_pairs, dmatch_threshold)
        except (AssertionError, AttributeError, IndexError) as e:
            sc_logger.warning(f"  → 批量匹配异常: {type(e).__name__}, 降级为空匹配")
            batch_results = [[] for _ in sc_pairs]
    else:
        batch_results = []
    for batch_idx, meta in enumerate(candidate_cluster_pairs):
        cluster_count += 1
        sc_id = meta['sc_id']
        q_rep = meta['q_rep']
        d_rep = meta['d_rep']
        q_text = meta['q_text']
        d_text = meta['d_text']
        q_triple_repr = meta['q_triple_repr']
        d_triple_repr = meta['d_triple_repr']
        q_vertices = meta['q_vertices']
        d_vertices = meta['d_vertices']
        q_vs = meta['q_vs']
        d_vs = meta['d_vs']
        q_edges = meta['q_edges']
        d_edges = meta['d_edges']
        sim_score = meta['sim_score']
        if batch_idx < len(batch_results):
            matches = {
                (vertex_to_sim_id_q[vq], vertex_to_sim_id_d[vd])
                for vq, vd, _ in batch_results[batch_idx]
                if vq in vertex_to_sim_id_q and vd in vertex_to_sim_id_d
            }
        else:
            matches = set()
        d_delta_matches[(sc_id, sc_id)] = matches
        sc_logger.info(
            f"→ 采纳 #{cluster_count} | score={sim_score:.3f}\n"
            f"  Q_rep=Q{q_rep.id}('{q_rep.text()}')\n"
            f"     full_text='{q_text}'\n"
            f"     triple={q_triple_repr}\n"
            f"     nodes={len(q_vertices)} (noun={len(q_vs)}), edges={len(q_edges)}\n"
            f"  D_rep=D{d_rep.id}('{d_rep.text()}')\n"
            f"     full_text='{d_text}'\n"
            f"     triple={d_triple_repr}\n"
            f"     nodes={len(d_vertices)} (noun={len(d_vs)}), edges={len(d_edges)}\n"
            f"  D-Match count: {len(matches)}"
        )
    sc_logger.info(f"语义簇构建完成: 原始 {len(raw_pairs)} → 有效 {cluster_count} 个簇对")   
    time2 = time.time()
    return delta, DMatch.from_dict(d_delta_matches)
def compute_hyper_simulation(
    query_hg: LocalHypergraph,
    data_hg: LocalHypergraph,
    sigma_threshold: float = 0.75,
    b_threshold: int = 5,
    delta_threshold: float = 0.7,
) -> Tuple[Dict[int, Set[int]], Dict[int, Vertex], Dict[int, Vertex]]:
    sim_logger = getLogger("hyper_simulation")
    sim_logger.debug(f"\tStart Hyper Simulation")
    q_sim, q_texts, q_vertices, q_edges, q_vid_map = convert_local_to_sim(query_hg)
    d_sim, d_texts, d_vertices, d_edges, d_vid_map = convert_local_to_sim(data_hg)
    denial_start = time.time()
    sim_logger.debug(f"\tstart denial comment calc")
    dc_logger = getLogger("denial_comment")
    time1 = time.time()
    allowed, confidence_scores = compute_allowed_pairs_batch_with_score(q_vertices, d_vertices)
    time2 = time.time()
    q_vertices_list = list(q_vertices.values())
    d_vertices_list = list(d_vertices.values())
    time3 = time.time()
    match_vertices = get_top_k_matched_vertices_by_scores(q_vertices, d_vertices, confidence_scores, k=b_threshold)
    def type_same_fn(x_id: int, y_id: int) -> bool:
        return (x_id, y_id) in allowed
    q_sim.set_type_same_fn(type_same_fn)
    d_sim.set_type_same_fn(type_same_fn)
    denial_end = time.time()
    dc_logger.info(f"\tdenial comment cost {denial_end - denial_start}s")
    sim_logger.debug(f"\tdenial comment cost {denial_end - denial_start}s")
    sim_logger.debug(f"\tstart build delta and d-match")
    delta, d_match = build_delta_and_dmatch(
        q_sim, d_sim, q_texts, d_texts, q_edges, d_edges, allowed,
        query_local_hg=query_hg,
        data_local_hg=data_hg,
        vertex_to_sim_id_q=q_vid_map,
        vertex_to_sim_id_d=d_vid_map,
        matched_vertices=match_vertices,
        dmatch_threshold=delta_threshold,
        cluster_sim_threshold=sigma_threshold,
        branch_threshold=b_threshold
    )
    start_time = time.time()
    sim_logger.info("\t执行超图模拟...")
    simulation = SimHypergraph.get_hyper_simulation(q_sim, d_sim, delta, d_match)
    sim_logger.info("\t=== Hyper Simulation Mapping ===")
    for q_id, d_ids in sorted(simulation.items()):
        q_text = q_vertices[q_id].text() if q_id in q_vertices else f"[Q{q_id}]"
        if d_ids:
            d_items = []
            for d_id in sorted(d_ids):
                if d_id in d_vertices:
                    d_text = d_vertices[d_id].text()
                    d_items.append(f"D{d_id}: '{d_text}'")
                else:
                    d_items.append(f"D{d_id}")
            targets = ", ".join(d_items)
        else:
            targets = "-"
        sim_logger.info(f"\t  Q{q_id}: '{q_text}' → {targets}")
    sim_logger.info("\t================================")
    end_time = time.time()
    sim_logger.info(f"\t模拟完成: {len(simulation)}个映射")
    sim_logger.info(f"\thyper simulation main cost {end_time - start_time}s")
    return simulation, q_vertices, d_vertices
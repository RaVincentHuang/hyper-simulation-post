"""Dependency-free contract tests for the HyperMatch core."""

from __future__ import annotations

from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout
import ast
import io
import os
import re
import subprocess
import sys
import tokenize
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hyper_simulation.component.config import (
    FANCY_CONFIG,
    STANDARD_CONFIG,
    HCCalcConfig,
    config_for_mode,
    config_summary,
)
from hyper_simulation.component.consistent import (
    MatchedHCWitness,
    compute_complete_cores,
    compute_query_consistency,
    mark_documents,
)
from hyper_simulation.component.d_match import (
    CallableRoleCellScorer,
    RoleScoringResult,
    build_role_cells,
    compute_dmatch,
)
from hyper_simulation.component.fixed_point import compute_hyper_simulation
from hyper_simulation.component.semantic_cluster import enumerate_hc_candidates
from hyper_simulation.component.hyper_simulation import run_hypermatch
from hyper_simulation.component.contracts import (
    Cluster,
    DMatchDecision,
    HCCandidate,
    HCDependency,
    HCFailure,
    HCRegistry,
    Hyperedge,
    Hypergraph,
    HyperSimulationResult,
    Removal,
    Role,
    Vertex,
)
from hyper_simulation.component.scoring import render_cluster


def predicate(id_: str, text: str) -> Vertex:
    return Vertex(id_, text, "PREDICATE", "predicate")


def entity(
    id_: str,
    text: str,
    type_: str,
    kind: str = "entity",
    *,
    expected_type: str | None = None,
    referent_id: str | None = None,
) -> Vertex:
    return Vertex(id_, text, type_, kind, expected_type, referent_id)


def edge(
    id_: str,
    predicate_id: str,
    context_id: str,
    roles: dict[str, str],
    *,
    frame: str = "",
    father_id: str | None = None,
    time: str | None = None,
    quantity: str | None = None,
) -> Hyperedge:
    return Hyperedge(
        id_,
        predicate_id,
        tuple(Role(name, vertex_id) for name, vertex_id in roles.items()),
        context_id,
        canonical_frame=frame,
        time=time,
        quantity=quantity,
        father_id=father_id,
    )


def two_hop_graphs(*, reverse: bool = False) -> tuple[Hypergraph, Hypergraph]:
    query_vertices = [
        predicate("qp1", "born"),
        predicate("qp2", "located in"),
        entity("q_person", "Alice", "PERSON"),
        entity("q_place", "?birthplace", "LOC", "query", expected_type="LOC"),
        entity("q_country", "?country", "GPE", "query", expected_type="GPE"),
    ]
    query_edges = [
        edge(
            "qe1",
            "qp1",
            "query",
            {"person": "q_person", "place": "q_place"},
            frame="birth",
        ),
        edge(
            "qe2",
            "qp2",
            "query",
            {"located": "q_place", "region": "q_country"},
            frame="location",
        ),
    ]
    data_vertices = [
        predicate("dp1", "was born"),
        predicate("dp2", "is located in"),
        predicate("dp3", "enjoys"),
        predicate("dp4", "is located in"),
        entity("d_alice", "Alice", "PERSON", referent_id="ref:alice"),
        entity("d_paris", "Paris", "LOC", referent_id="ref:paris-france"),
        entity("d_france", "France", "GPE", referent_id="ref:france"),
        entity("d_chess", "Chess", "PRODUCT", referent_id="ref:chess"),
        entity("d_paris_tx", "Paris", "LOC", referent_id="ref:paris-texas"),
        entity("d_texas", "Texas", "GPE", referent_id="ref:texas"),
    ]
    data_edges = [
        edge(
            "d1",
            "dp1",
            "c1",
            {"person": "d_alice", "place": "d_paris"},
            frame="birth",
        ),
        edge(
            "d2",
            "dp2",
            "c1",
            {"located": "d_paris", "region": "d_france"},
            frame="location",
        ),
        edge(
            "d3",
            "dp3",
            "c1",
            {"experiencer": "d_alice", "theme": "d_chess"},
            frame="enjoyment",
        ),
        edge(
            "d4",
            "dp4",
            "c2",
            {"located": "d_paris_tx", "region": "d_texas"},
            frame="location",
        ),
    ]
    if reverse:
        query_vertices.reverse()
        query_edges.reverse()
        data_vertices.reverse()
        data_edges.reverse()
    return (
        Hypergraph(tuple(query_vertices), tuple(query_edges), "query"),
        Hypergraph(tuple(data_vertices), tuple(data_edges), "data"),
    )


class HyperMatchCoreTests(unittest.TestCase):
    def test_public_flags_select_distinct_immutable_profiles(self) -> None:
        from hyper_simulation.question_answer import hypermatch as qa_entrypoint

        self.assertFalse(qa_entrypoint.build_parser().parse_args([]).fancy)
        self.assertTrue(
            qa_entrypoint.build_parser().parse_args(["--fancy"]).fancy
        )
        self.assertFalse(
            qa_entrypoint.build_parser().parse_args(["--no-fancy"]).fancy
        )
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                qa_entrypoint.build_parser().parse_args(
                    ["--fancy", "--no-fancy"]
                )
        self.assertIs(config_for_mode(fancy=False), STANDARD_CONFIG)
        self.assertIs(config_for_mode(fancy=True), FANCY_CONFIG)
        self.assertNotEqual(
            STANDARD_CONFIG.fingerprint, FANCY_CONFIG.fingerprint
        )
        self.assertEqual(
            STANDARD_CONFIG.fingerprint,
            "e6d42323d8c51f7b4e7d8f02a7b01eeb2c279cc29fe125cc90816e0dc13722c2",
        )
        self.assertEqual(
            FANCY_CONFIG.fingerprint,
            "e379651c9e9c1bd6f2cf503cba45ca1b341d3ffe964c68656222b94c1a0b8120",
        )
        for selected in (STANDARD_CONFIG, FANCY_CONFIG):
            summary = config_summary(selected)
            self.assertNotIn("version", summary)
            self.assertIn(summary["mode"], {"standard", "fancy"})

        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(qa_entrypoint.main(["--fancy"]), 0)
        self.assertIn('"mode": "fancy"', output.getvalue())

    def test_profile_names_and_documentation_describe_the_implementation(self) -> None:
        def documentation_text(path: Path) -> str:
            """Return prose, Python comments/docstrings, or Rust comments only."""

            source = path.read_text()
            if path.suffix == ".md":
                prose: list[str] = []
                inside_fence = False
                for line in source.splitlines():
                    if line.lstrip().startswith("```"):
                        inside_fence = not inside_fence
                    elif not inside_fence:
                        prose.append(line)
                return "\n".join(prose)
            if path.suffix == ".rs":
                line_comments = [
                    line.split("//", 1)[1]
                    for line in source.splitlines()
                    if "//" in line
                ]
                block_comments = re.findall(r"/\*.*?\*/", source, flags=re.DOTALL)
                return "\n".join((*line_comments, *block_comments))

            tree = ast.parse(source)
            docstrings = [
                value
                for node in ast.walk(tree)
                if isinstance(
                    node,
                    (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
                )
                if (value := ast.get_docstring(node, clean=False)) is not None
            ]
            comments = [
                token.string
                for token in tokenize.generate_tokens(io.StringIO(source).readline)
                if token.type == tokenize.COMMENT
            ]
            return "\n".join((*docstrings, *comments))

        paths = [
            ROOT / "README.md",
            *(ROOT / "src" / "hyper_simulation").rglob("*.py"),
            *(ROOT / "src" / "graph-simulation").rglob("*.rs"),
            *(ROOT / "src" / "simulation").rglob("*.rs"),
        ]
        forbidden = (
            "HyperMatch " + "1.0",
            "HyperMatch " + "2.0",
            "hypermatch_" + "v1",
            "hypermatch_" + "v2",
            "get_" + "v1_symbol",
            "_compute_hyper_simulation_" + "v1",
            "_load_" + "v1_dependencies",
            "review" + "er",
            "08" + "15",
            "paper_" + "all_hc",
            "Appen" + "dix E",
            "Section " + "4.",
            "Algorithm " + "2",
            "Definition " + "4.",
            "2" + "(a)",
            "2" + "(b)",
            "fancy profile",
            "standard profile",
            "standard branch",
            "standard path",
            "standard solver",
            "standard implementation",
            "standard adapter",
            "profile dispatch",
            "standard/fancy",
            "fancy",
            "--" + "fancy",
            "--" + "no-fancy",
        )
        for path in paths:
            source = documentation_text(path).casefold()
            for marker in forbidden:
                self.assertNotIn(
                    marker.casefold(), source, f"{marker!r} in {path.name}"
                )

    def test_component_is_library_only_and_qa_owns_execution(self) -> None:
        from hyper_simulation.component import hyper_simulation as library
        from hyper_simulation.question_answer import hypermatch as qa_entrypoint

        component_dir = ROOT / "src" / "hyper_simulation" / "component"
        self.assertFalse((component_dir / "__main__.py").exists())
        self.assertFalse(hasattr(library, "build_parser"))
        self.assertFalse(hasattr(library, "main"))
        self.assertTrue(hasattr(library, "compute_hyper_simulation"))
        self.assertTrue(hasattr(library, "run_hypermatch"))
        self.assertTrue(hasattr(library, "run_selected"))

        for source_path in component_dir.glob("*.py"):
            tree = ast.parse(source_path.read_text())
            for node in tree.body:
                if not isinstance(node, ast.If):
                    continue
                rendered = ast.unparse(node.test)
                self.assertNotIn("__main__", rendered, source_path.name)

        pyproject = (ROOT / "pyproject.toml").read_text()
        self.assertIn(
            'hypermatch = "hyper_simulation.question_answer.hypermatch:main"',
            pyproject,
        )
        self.assertIn(
            'hypermatch = "python -m hyper_simulation.question_answer.hypermatch"',
            pyproject,
        )

        self.assertEqual(qa_entrypoint.__all__, ["build_parser", "main"])
        self.assertFalse(hasattr(qa_entrypoint, "run_selected"))
        self.assertFalse(hasattr(qa_entrypoint, "query_fixup"))

        rag_source = (
            ROOT
            / "src"
            / "hyper_simulation"
            / "question_answer"
            / "rag_no_retrival.py"
        ).read_text()
        self.assertIn(
            "from hyper_simulation.component.hyper_simulation import query_fixup",
            rag_source,
        )
        self.assertNotIn(
            "from hyper_simulation.question_answer.hypermatch import",
            rag_source,
        )

    def test_dispatch_is_lazy_and_defaults_to_the_standard_profile(self) -> None:
        from hyper_simulation.component import hyper_simulation as entrypoint

        with (
            patch.object(entrypoint, "_load_standard_dependencies") as loader,
            patch.object(
                entrypoint,
                "_compute_standard_hyper_simulation",
                return_value=("standard", ("query", "data"), {}),
            ) as run_standard,
        ):
            value = entrypoint.compute_hyper_simulation("query", "data")
        self.assertEqual(value[0], "standard")
        self.assertEqual(value[1], ("query", "data"))
        self.assertEqual(value[2], {})
        loader.assert_called_once_with()
        run_standard.assert_called_once_with("query", "data")

        with (
            patch.object(entrypoint, "_load_standard_dependencies") as loader,
            patch.object(
                entrypoint,
                "_compute_standard_hyper_simulation",
                return_value=("standard", (), {"sigma_threshold": 0.75}),
            ) as run_standard,
        ):
            value = entrypoint.compute_hyper_simulation(
                "query", "data", fancy=False, sigma_threshold=0.75
            )
        self.assertEqual(value[2], {"sigma_threshold": 0.75})
        loader.assert_called_once_with()
        run_standard.assert_called_once_with(
            "query", "data", sigma_threshold=0.75
        )

        with (
            patch.object(
                entrypoint, "run_hypermatch", return_value="fancy"
            ) as run_fancy,
            patch.object(entrypoint, "_load_standard_dependencies") as loader,
            patch.object(
                entrypoint, "_compute_standard_hyper_simulation"
            ) as run_standard,
        ):
            self.assertEqual(
                entrypoint.compute_hyper_simulation(
                    "query", "data", fancy=True, hv_allowed_pairs=()
                ),
                "fancy",
            )
        run_fancy.assert_called_once()
        loader.assert_not_called()
        run_standard.assert_not_called()

    def test_standard_solvers_are_inlined_in_public_module(self) -> None:
        from hyper_simulation.component import consistent as consistency_facade
        from hyper_simulation.component import d_match as dmatch_facade
        from hyper_simulation.component import hyper_simulation as simulation_facade
        from hyper_simulation.component import semantic_cluster as hc_facade

        module_name = "hyper_simulation.component.hyper_simulation"
        component_dir = ROOT / "src" / "hyper_simulation" / "component"
        private_fragments = [
            path
            for path in component_dir.glob("_*.py")
            if path.name != "__init__.py"
        ]
        self.assertEqual([], private_fragments)

        tree = ast.parse((component_dir / "hyper_simulation.py").read_text())
        definitions = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertTrue(
            {
                "SemanticCluster",
                "calc_semantic_cluster_pairs",
                "get_d_match",
                "calc_d_match",
                "calc_d_match_batch",
                "convert_local_to_sim",
                "build_delta_and_dmatch",
                "_compute_standard_hyper_simulation",
                "consistent_detection",
                "load_hypergraphs_for_instance",
                "query_fixup",
            }
            <= definitions
        )

        values = {
            "SemanticCluster": "semantic-cluster",
            "calc_semantic_cluster_pairs": lambda *args, **kwargs: "hc",
            "get_semantic_cluster_pairs": lambda *args, **kwargs: "hc-before-threshold",
            "get_d_match": lambda *args, **kwargs: "cluster-dmatch",
            "calc_d_match": lambda *args, **kwargs: "dmatch",
            "calc_d_match_batch": lambda *args, **kwargs: "dmatch-batch",
            "consistent_detection": lambda *args, **kwargs: "consistency",
            "load_hypergraphs_for_instance": lambda *args, **kwargs: "graphs",
            "query_fixup": lambda *args, **kwargs: "fixed-query",
        }

        checks = (
            (hc_facade, lambda: hc_facade.calc_semantic_cluster_pairs(), "hc"),
            (hc_facade, lambda: hc_facade.get_semantic_cluster_pairs(), "hc-before-threshold"),
            (hc_facade, lambda: hc_facade.get_d_match(), "cluster-dmatch"),
            (dmatch_facade, lambda: dmatch_facade.calc_d_match(), "dmatch"),
            (dmatch_facade, lambda: dmatch_facade.calc_d_match_batch(), "dmatch-batch"),
            (consistency_facade, lambda: consistency_facade.consistent_detection(), "consistency"),
            (consistency_facade, lambda: consistency_facade.load_hypergraphs_for_instance(), "graphs"),
            (consistency_facade, lambda: consistency_facade.query_fixup(), "fixed-query"),
        )
        for facade, invoke, expected in checks:
            with self.subTest(facade=facade.__name__, expected=expected):
                with patch.object(
                    simulation_facade,
                    "get_standard_symbol",
                    side_effect=lambda name: values[name],
                ) as resolver:
                    self.assertEqual(invoke(), expected)
                resolver.assert_called_once()

        with patch.object(
            simulation_facade,
            "get_standard_symbol",
            side_effect=lambda name: values[name],
        ) as resolver:
            self.assertEqual(hc_facade.SemanticCluster, "semantic-cluster")
        resolver.assert_called_once_with("SemanticCluster")

        for facade in (hc_facade, dmatch_facade, consistency_facade):
            with patch.object(
                facade.importlib, "import_module", return_value=simulation_facade
            ) as loader:
                self.assertIs(facade._solver_module(), simulation_facade)
            loader.assert_called_once_with(module_name)

        for facade in (hc_facade, dmatch_facade, consistency_facade):
            with patch.object(facade.importlib, "import_module") as loader:
                with self.assertRaises(AttributeError):
                    getattr(facade, "not_a_solver_symbol")
            loader.assert_not_called()

        with patch.object(
            simulation_facade, "_load_standard_dependencies"
        ) as loader:
            with self.assertRaises(AttributeError):
                simulation_facade.get_standard_symbol("not_a_solver_symbol")
        loader.assert_not_called()

    def test_fancy_import_does_not_load_the_standard_model_stack(self) -> None:
        code = """
import sys
from hyper_simulation.component import hyper_simulation
from hyper_simulation.question_answer import hypermatch
forbidden = {
    'torch', 'transformers', 'sentence_transformers', 'numpy', 'simulation',
    'sentencepiece', 'jsonlines', 'langchain_ollama',
    'hyper_simulation.question_answer.rag',
    'hyper_simulation.question_answer.rag_no_retrival',
}
loaded = sorted(forbidden & set(sys.modules))
assert not loaded, loaded
assert not hasattr(hyper_simulation, 'main')
assert hypermatch.main(['--fancy']) == 0
loaded = sorted(forbidden & set(sys.modules))
assert not loaded, loaded
"""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT / "src")
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )

        for flag, expected_mode in (
            ("--fancy", "fancy"),
            ("--no-fancy", "standard"),
        ):
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "hyper_simulation.question_answer.hypermatch",
                    flag,
                ],
                cwd=ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn(f'"mode": "{expected_mode}"', completed.stdout)

    def test_standard_types_import_without_models_or_writable_logs(self) -> None:
        code = """
from hyper_simulation.hypergraph.hypergraph import Hypergraph
from hyper_simulation.component.semantic_cluster import SemanticCluster
from hyper_simulation.component import hyper_simulation
from hyper_simulation.utils.log import getLogger
assert Hypergraph.__name__ == 'Hypergraph'
assert SemanticCluster.__name__ == 'SemanticCluster'
assert not hyper_simulation._STANDARD_DEPENDENCIES_LOADED
logger = getLogger('readonly-test', log_dir='/proc/hyper-simulation-logs')
assert logger.handlers
"""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT / "src")
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        subprocess.run(
            [sys.executable, "-c", code],
            cwd="/proc",
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_masking_hides_entity_values_but_preserves_lexical_structure(self) -> None:
        vertices = (
            predicate("p", "sold"),
            entity("alice", "Alice", "PERSON"),
            entity("acme", "Acme", "ORG"),
            entity("buyer", "?buyer", "PERSON", "query", expected_type="PERSON"),
            entity("quickly", "quickly", "MANNER", "adverb"),
        )
        graph = Hypergraph(
            vertices,
            (
                edge(
                    "e",
                    "p",
                    "query",
                    {
                        "agent": "alice",
                        "theme": "acme",
                        "recipient": "buyer",
                        "manner": "quickly",
                    },
                    frame="transfer",
                    time="April 3 1999",
                    quantity="$5 million",
                ),
            ),
            "query",
        )
        rendered = render_cluster(
            graph, Cluster.from_edges(graph, ("e",), context_id="query"), view="fine"
        )
        self.assertNotIn("alice", rendered.casefold())
        self.assertNotIn("acme", rendered.casefold())
        self.assertNotIn("?buyer", rendered)
        self.assertNotIn("april", rendered.casefold())
        self.assertNotIn("million", rendered.casefold())
        self.assertIn("time=<TEMPORAL>", rendered)
        self.assertIn("quantity=<QUANTITY>", rendered)
        self.assertIn("PERSON#", rendered)
        self.assertIn("ORG#", rendered)
        self.assertIn("sold", rendered)
        self.assertIn("quickly", rendered)
        self.assertIn("agent=", rendered)
        self.assertIn("recipient=", rendered)

    def test_hccalc_is_deterministic_and_context_local(self) -> None:
        query, data = two_hop_graphs()
        scores = {
            ("qe1", "d1"): 0.95,
            ("qe1", "d2"): 0.15,
            ("qe1", "d3"): 0.20,
            ("qe1", "d4"): 0.10,
            ("qe2", "d1"): 0.15,
            ("qe2", "d2"): 0.94,
            ("qe2", "d3"): 0.10,
            ("qe2", "d4"): 0.99,
        }
        hv = {
            ("q_person", "d_alice"),
            ("q_place", "d_paris"),
            ("q_place", "d_paris_tx"),
            ("q_country", "d_france"),
            ("q_country", "d_texas"),
        }
        config = HCCalcConfig(seed_edges=2, structural_beam=7)
        first = enumerate_hc_candidates(query, data, hv, scores, config)
        query_reversed, data_reversed = two_hop_graphs(reverse=True)
        second = enumerate_hc_candidates(
            query_reversed,
            data_reversed,
            reversed(sorted(hv)),
            dict(reversed(list(scores.items()))),
            config,
        )
        self.assertEqual(first, second)
        self.assertEqual(len({candidate.id for candidate in first}), len(first))
        self.assertTrue(
            any(
                set(candidate.query_cluster.edge_ids) == {"qe1", "qe2"}
                and set(candidate.data_cluster.edge_ids) == {"d1", "d2"}
                for candidate in first
            )
        )
        for candidate in first:
            contexts = {data.edge(edge_id).context_id for edge_id in candidate.data_cluster.edge_ids}
            self.assertEqual(len(contexts), 1)
            self.assertFalse({"d1", "d4"} <= set(candidate.data_cluster.edge_ids))
            if "father_atom_incidence_closure" not in candidate.routes:
                self.assertLessEqual(
                    len(candidate.data_cluster.edge_ids), config.max_cluster_size
                )

    def test_dmatch_hard_types_and_semantic_one_to_one_decode(self) -> None:
        query = Hypergraph(
            (
                predicate("sell", "sells"),
                entity("q_a", "A", "PERSON"),
                entity("q_b", "B", "PERSON"),
                entity("q_item", "item", "PRODUCT"),
            ),
            (
                edge(
                    "qe",
                    "sell",
                    "query",
                    {"source": "q_a", "theme": "q_item", "goal": "q_b"},
                    frame="transfer",
                ),
            ),
            "query",
        )
        data = Hypergraph(
            (
                predicate("buy", "buys"),
                entity("d_c", "C", "PERSON"),
                entity("d_d", "D", "PERSON"),
                entity("d_item", "item", "PRODUCT"),
                entity("d_time", "yesterday", "TEMPORAL", "value"),
                entity("d_virtual", "implicit", "UNKNOWN", "virtual"),
            ),
            (
                edge(
                    "de",
                    "buy",
                    "c1",
                    {
                        "source": "d_d",
                        "theme": "d_item",
                        "goal": "d_c",
                        "time": "d_time",
                        "implicit": "d_virtual",
                    },
                    frame="transfer",
                ),
            ),
            "data",
        )
        candidate = HCCandidate(
            Cluster.from_edges(query, ("qe",), context_id="query"),
            Cluster.from_edges(data, ("de",), context_id="c1"),
            ("relation_seeded",),
            1.0,
        )
        hv = {
            ("q_a", "d_d"),
            ("q_a", "d_c"),
            ("q_b", "d_c"),
            ("q_b", "d_d"),
            ("q_item", "d_item"),
            ("q_item", "d_time"),  # deliberately incompatible
        }
        cells = build_role_cells(query, data, candidate, hv_allowed_pairs=hv)
        legal = {cell.pair for cell in cells}
        self.assertNotIn(("q_item", "d_time"), legal)
        self.assertTrue(all("sell" not in pair and "buy" not in pair for pair in legal))
        self.assertTrue(all("d_virtual" not in pair for pair in legal))

        desired = {
            ("q_a", "d_d"): 0.99,
            ("q_b", "d_c"): 0.98,
            ("q_item", "d_item"): 0.97,
            ("q_a", "d_c"): 0.40,
            ("q_b", "d_d"): 0.39,
        }
        scorer = CallableRoleCellScorer(
            lambda values: RoleScoringResult(
                {cell.pair: desired.get(cell.pair, 0.0) for cell in values}
            )
        )
        result = compute_dmatch(
            query,
            data,
            candidate,
            scorer,
            hv_allowed_pairs=hv,
            anchor_pairs={
                ("q_a", "d_d"),
                ("q_b", "d_c"),
                ("q_item", "d_item"),
            },
            threshold=0.5,
        )
        self.assertEqual(
            result.pairs,
            frozenset(
                {
                    ("q_a", "d_d"),
                    ("q_b", "d_c"),
                    ("q_item", "d_item"),
                }
            ),
        )

    def test_inference_error_abstains_but_conflict_is_empty(self) -> None:
        query, data = two_hop_graphs()
        candidate = HCCandidate(
            Cluster.from_edges(query, ("qe2",), context_id="query"),
            Cluster.from_edges(data, ("d2",), context_id="c1"),
            ("relation_seeded",),
            1.0,
        )
        # All four pairs pass the broad location type gate.  They are not a
        # partial one-to-one relation and therefore must never be fabricated
        # into D-match when the scorer fails.
        anchors = {
            ("q_place", "d_paris"),
            ("q_place", "d_france"),
            ("q_country", "d_paris"),
            ("q_country", "d_france"),
        }

        def fail(_cells):
            raise RuntimeError("synthetic inference failure")

        result = compute_dmatch(
            query,
            data,
            candidate,
            CallableRoleCellScorer(fail),
            hv_allowed_pairs=anchors,
            anchor_pairs=anchors,
            threshold=0.5,
        )
        self.assertEqual(result.status, "inference_error_abstain")
        self.assertEqual(result.effective_pairs, frozenset())
        with self.assertRaises(ValueError):
            HCDependency(
                "failed-hc",
                candidate.query_cluster,
                candidate.data_cluster,
                frozenset(anchors),
                result,
                "full",
            )

        conflict = compute_dmatch(
            query,
            data,
            candidate,
            CallableRoleCellScorer(
                lambda _cells: RoleScoringResult({}, relation_conflict=True)
            ),
            hv_allowed_pairs=anchors,
            anchor_pairs=anchors,
            threshold=0.5,
        )
        self.assertTrue(conflict.relation_conflict)
        self.assertEqual(conflict.effective_pairs, frozenset())

    def test_one_logical_hc_is_registered_once_for_multiple_anchors(self) -> None:
        query, data = two_hop_graphs()
        q_cluster = Cluster.from_edges(query, ("qe1",), context_id="query")
        d_cluster = Cluster.from_edges(data, ("d1",), context_id="c1")
        anchors = frozenset({("q_person", "d_alice"), ("q_place", "d_paris")})
        dependency = HCDependency(
            "logical-hc",
            q_cluster,
            d_cluster,
            anchors,
            DMatchDecision(anchors),
            "full",
        )
        registry = HCRegistry()
        registry.register(dependency)
        registry.register(
            HCDependency(
                "logical-hc",
                q_cluster,
                d_cluster,
                frozenset(reversed(sorted(anchors))),
                DMatchDecision(frozenset(reversed(sorted(anchors)))),
                "full",
            )
        )
        self.assertEqual(len(registry), 1)
        for anchor in anchors:
            self.assertEqual(registry.ids_for_anchor(anchor), ("logical-hc",))

    def test_fixed_point_cascade_uses_parallel_iterations(self) -> None:
        cluster_q = Cluster(("qe",), "query")
        cluster_d = Cluster(("de",), "c1")
        p_a = ("qA", "dA")
        p_b = ("qB", "dB")
        p_c = ("qC", "dC")
        dependencies = (
            HCDependency(
                "hcA",
                cluster_q,
                cluster_d,
                frozenset({p_a}),
                DMatchDecision(frozenset({p_a, p_b})),
                "full",
            ),
            HCDependency(
                "hcB",
                cluster_q,
                cluster_d,
                frozenset({p_b}),
                DMatchDecision(frozenset({p_b, p_c})),
                "full",
            ),
            HCDependency(
                "hcC",
                cluster_q,
                cluster_d,
                frozenset({p_c}),
                DMatchDecision(frozenset()),
                "full",
            ),
        )
        result = compute_hyper_simulation({p_a, p_b, p_c}, dependencies)
        self.assertEqual(result.relation, frozenset())
        self.assertEqual(result.iterations, 3)
        self.assertEqual(
            [(item.iteration, item.pair, item.reason) for item in result.removals],
            [
                (1, p_c, "empty_dmatch"),
                (2, p_b, "dmatch_not_closed"),
                (3, p_a, "dmatch_not_closed"),
            ],
        )
        self.assertEqual(
            [item.failures[0].condition for item in result.removals],
            ["anchor_membership", "dependency_closure", "dependency_closure"],
        )

        fail_open = dependencies[:-1] + (
            HCDependency(
                "hcC",
                cluster_q,
                cluster_d,
                frozenset({p_c}),
                DMatchDecision(frozenset({p_c}), status="manual_conservative"),
                "full",
            ),
        )
        retained = compute_hyper_simulation({p_a, p_b, p_c}, fail_open)
        self.assertEqual(retained.relation, frozenset({p_a, p_b, p_c}))

        support_only = HCDependency(
            "support",
            cluster_q,
            cluster_d,
            frozenset({p_a}),
            DMatchDecision(frozenset()),
            "support_only",
        )
        self.assertEqual(
            compute_hyper_simulation({p_a}, (support_only,)).relation,
            frozenset({p_a}),
        )

    def test_query_coverage_and_minimal_complete_cores(self) -> None:
        consistency = compute_query_consistency(
            ("q_person", "q_bridge", "q_answer"),
            {
                ("q_person", "d_person"),
                ("q_bridge", "d_bridge"),
                ("q_answer", "d_answer"),
            },
        )
        self.assertTrue(consistency.consistent)
        missing = compute_query_consistency(
            ("q_person", "q_bridge", "q_answer", "q_missing"),
            {
                ("q_person", "d_person"),
                ("q_bridge", "d_bridge"),
                ("q_answer", "d_answer"),
            },
        )
        self.assertFalse(missing.consistent)
        self.assertEqual(missing.uncovered_query_vertex_ids, ("q_missing",))

        w1 = MatchedHCWitness(
            "w1",
            ("qe1",),
            ("d1",),
            "c1",
            (("q_person", "d_person"), ("q_bridge", "d_bridge_a")),
        )
        w2 = MatchedHCWitness(
            "w2",
            ("qe2",),
            ("d2",),
            "c2",
            (("q_bridge", "d_bridge_b"), ("q_answer", "d_answer")),
        )
        result = compute_complete_cores(
            ("qe1", "qe2"),
            ("q_person", "q_bridge", "q_answer"),
            (w1, w2),
        )
        # Complete-core coverage takes unions over the selected witnesses.  It
        # imposes no extra connectivity or cross-HC single-binding condition,
        # so the two q_bridge targets do not invalidate this formal core.
        self.assertTrue(result.complete)
        self.assertEqual(result.selected_core.context_ids, ("c1", "c2"))

        missing_vertex = compute_complete_cores(
            ("qe1", "qe2"),
            ("q_person", "q_bridge", "q_answer", "q_unmatched"),
            (w1, w2),
        )
        self.assertFalse(missing_vertex.complete)
        self.assertEqual(
            missing_vertex.uncovered_query_vertex_ids, ("q_unmatched",)
        )

        direct = MatchedHCWitness(
            "w3",
            ("qe1", "qe2"),
            ("d3",),
            "c3",
            (
                ("q_person", "d_person_3"),
                ("q_bridge", "d_bridge_3"),
                ("q_answer", "d_answer_3"),
            ),
        )
        alternatives = compute_complete_cores(
            ("qe1", "qe2"),
            ("q_person", "q_bridge", "q_answer"),
            (w1, w2, direct),
        )
        self.assertEqual(
            {core.witness_ids for core in alternatives.minimal_cores},
            {("w1", "w2"), ("w3",)},
        )

    def test_document_marking_uses_complete_core_conflict_certificate_and_local_relation(self) -> None:
        query, data = two_hop_graphs()
        c1_dependency = HCDependency(
            "hc-core",
            Cluster.from_edges(query, ("qe1",), context_id="query"),
            Cluster.from_edges(data, ("d1",), context_id="c1"),
            frozenset({("q_person", "d_alice")}),
            DMatchDecision(frozenset({("q_person", "d_alice")})),
            "full",
            ("qe1",),
        )
        c2_dependency = HCDependency(
            "hc-conflict",
            Cluster.from_edges(query, ("qe2",), context_id="query"),
            Cluster.from_edges(data, ("d4",), context_id="c2"),
            frozenset({("q_place", "d_paris_tx")}),
            DMatchDecision(frozenset(), relation_conflict=True),
            "full",
            ("qe2",),
        )
        fixed = HyperSimulationResult(
            initial_relation=frozenset(
                {
                    ("q_person", "d_alice"),
                    ("q_place", "d_paris"),
                    ("q_country", "d_france"),
                    ("q_place", "d_paris_tx"),
                }
            ),
            relation=frozenset(
                {
                    ("q_person", "d_alice"),
                    ("q_place", "d_paris"),
                    ("q_country", "d_france"),
                }
            ),
            iterations=1,
            removals=(
                Removal(
                    1,
                    ("q_place", "d_paris_tx"),
                    (
                        HCFailure(
                            "hc-conflict", "anchor_membership", "empty_dmatch"
                        ),
                    ),
                ),
            ),
        )
        core_witness = MatchedHCWitness(
            "hc-core",
            ("qe1", "qe2"),
            ("d1", "d2"),
            "c1",
            (
                ("q_person", "d_alice"),
                ("q_place", "d_paris"),
                ("q_country", "d_france"),
            ),
        )
        cores = compute_complete_cores(
            ("qe1", "qe2"),
            ("q_person", "q_place", "q_country"),
            (core_witness,),
        )
        marks = mark_documents(
            ("c1", "c2", "c3", "c4"),
            context_vertex_ids={
                "c1": {"d_alice", "d_paris", "d_france"},
                "c2": {"d_paris_tx", "d_texas"},
                "c3": {"d_france"},
                # The deleted endpoint also occurs in c4 after fusion.  The
                # The conflict certificate belongs to the HC's source context
                # c2; endpoint overlap alone must not mark c4 as conflict.
                "c4": {"d_paris_tx"},
            },
            fixed_point=fixed,
            complete_cores=cores,
            dependencies=(c1_dependency, c2_dependency),
        )
        labels = {value.context_id: value.category for value in marks}
        self.assertEqual(
            labels,
            {
                "c1": "consistent",
                "c2": "conflict",
                "c3": "useless",
                "c4": "irrelevant",
            },
        )

    def test_schema_rejects_root_only_and_cross_context_clusters(self) -> None:
        with self.assertRaises(ValueError):
            Hyperedge("bad", "p", (), "c1")
        _, data = two_hop_graphs()
        with self.assertRaises(ValueError):
            Cluster.from_edges(data, ("d1", "d4"), context_id="c1")

    def test_dmatch_rejects_non_finite_scores(self) -> None:
        from hyper_simulation.component.d_match import stable_partial_one_to_one

        for score in (float("nan"), float("inf"), -0.1, 1.1):
            with self.assertRaises(ValueError):
                stable_partial_one_to_one({("q", "d"): score}, threshold=0.5)

    def test_end_to_end_reference_pipeline_needs_no_model_package(self) -> None:
        query, data = two_hop_graphs()

        class ToyBackend:
            def encode(self, texts, *, prompt=None):
                del prompt
                return tuple(
                    (0.0, 1.0) if "enjoyment" in text else (1.0, 0.0)
                    for text in texts
                )

        def score_roles(cells):
            scores = {}
            for cell in cells:
                query_roles = set(cell.query_roles)
                data_roles = set(cell.data_roles)
                scores[cell.pair] = 1.0 if query_roles & data_roles else 0.0
            return RoleScoringResult(scores)

        hv = {
            ("q_person", "d_alice"),
            ("q_place", "d_paris"),
            ("q_country", "d_france"),
            ("q_place", "d_paris_tx"),
            ("q_country", "d_texas"),
        }
        output = run_hypermatch(
            query,
            data,
            hv_allowed_pairs=hv,
            hc_backend=ToyBackend(),
            role_scorer=CallableRoleCellScorer(score_roles),
            query_clusters=(
                Cluster.from_edges(
                    query, ("qe1", "qe2"), context_id="query"
                ),
            ),
        )
        self.assertTrue(output.query_consistency.consistent)
        self.assertTrue(output.complete_cores.complete)
        labels = {value.context_id: value.label for value in output.document_marks}
        self.assertEqual(labels["c1"], "consistent")

        partial = run_hypermatch(
            query,
            data,
            hv_allowed_pairs=hv,
            hc_backend=ToyBackend(),
            role_scorer=CallableRoleCellScorer(
                lambda cells: RoleScoringResult(
                    {
                        cell.pair: (
                            1.0
                            if cell.pair == ("q_person", "d_alice")
                            else 0.0
                        )
                        for cell in cells
                    }
                )
            ),
            query_clusters=(
                Cluster.from_edges(
                    query, ("qe1", "qe2"), context_id="query"
                ),
            ),
        )
        self.assertFalse(partial.query_consistency.consistent)
        self.assertFalse(partial.complete_cores.complete)
        self.assertNotIn(
            "consistent", {value.label for value in partial.document_marks}
        )

        def fail_roles(_cells):
            raise RuntimeError("synthetic backend failure")

        abstained = run_hypermatch(
            query,
            data,
            hv_allowed_pairs=hv,
            hc_backend=ToyBackend(),
            role_scorer=CallableRoleCellScorer(fail_roles),
            query_clusters=(
                Cluster.from_edges(
                    query, ("qe1", "qe2"), context_id="query"
                ),
            ),
        )
        self.assertEqual(abstained.dependencies, ())
        self.assertEqual(abstained.fixed_point.relation, frozenset(hv))
        self.assertTrue(abstained.query_consistency.consistent)
        self.assertFalse(abstained.complete_cores.complete)


if __name__ == "__main__":
    unittest.main(verbosity=2)

from hyper_simulation.hypergraph.dependency import LocalDoc
from hyper_simulation.hypergraph.hypergraph import Hypergraph, Vertex
from dataclasses import dataclass
@dataclass
class Edge:
    src: Vertex
    dst: Vertex
    label: str
class Graph:
    def __init__(self, vertices: list[Vertex], edges: list[Edge], doc: LocalDoc) -> None:
        self.vertices = vertices
        self.edges = edges
        self.doc = doc
    @classmethod
    def from_hypergraph(cls, hypergraph: Hypergraph) -> 'Graph':
        vertices = hypergraph.vertices
        edges: list[Edge] = []
        for he in hypergraph.hyperedges:
            if not he.vertices:
                continue
            src = he.vertices[0]
            for dst in he.vertices[1:]:
                label = dst.dep().name if dst.dep().name else "dep"
                edges.append(Edge(src=src, dst=dst, label=label))
        return cls(vertices=vertices, edges=edges, doc=hypergraph.doc)
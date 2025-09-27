from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True, slots=True)
class SearchHit:
    doc_id: str
    score: float


@dataclass(slots=True)
class ServerAction:
    node: str
    server: str
    op: str
    detail: dict[str, object]


@dataclass(slots=True)
class ServerLog:
    actions: list[ServerAction] = field(default_factory=list)


@runtime_checkable
class SearchServer(Protocol):
    @property
    def name(self) -> str: ...

    def ensure(self, doc_ids: Iterable[str], *, from_nodes: Sequence[str]) -> None:
        pass

    def search(self, query: str, *, candidates: Iterable[str] | None = None, topk: int = 10) -> list[SearchHit]:
        pass


@dataclass(frozen=True, slots=True)
class NodeOutput:
    name: str
    hits: list[SearchHit]


@dataclass(slots=True)
class SearchNodeCfg:
    name: str
    server: str
    query: str
    topk: int = 10
    select_from: list[str] = field(default_factory=list)
    materialize_from: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MergeNodeCfg:
    name: str
    sources: list[str]
    mode: str = "union"
    topk: int | None = None
    score_norm: bool = True
    combiner: Callable[[list[float]], float] | None = None


NodeCfg = tuple[str, object]


class Workflow:
    def __init__(self) -> None:
        self._servers: dict[str, SearchServer] = {}
        self._nodes: list[NodeCfg] = []
        self._outputs: dict[str, NodeOutput] = {}
        self.log = ServerLog()

    def register_server(self, server: SearchServer) -> None:
        if server.name in self._servers:
            raise ValueError(f"server '{server.name}' already registered")
        self._servers[server.name] = server

    def add_search(self, **kwargs) -> None:
        cfg = SearchNodeCfg(**kwargs)
        self._nodes.append(("search", cfg))

    def add_merge(self, **kwargs) -> None:
        cfg = MergeNodeCfg(**kwargs)
        self._nodes.append(("merge", cfg))

    def run(self) -> dict[str, NodeOutput]:
        for kind, cfg in self._nodes:
            match kind:
                case "search":
                    out = self._exec_search(cfg)  # type: ignore[arg-type]
                case "merge":
                    out = self._exec_merge(cfg)  # type: ignore[arg-type]
                case _:
                    raise ValueError(kind)
            self._outputs[out.name] = out
        return self._outputs

    def report(self) -> str:
        lines: list[str] = ["===== Node Outputs ====="]
        for name, out in self._outputs.items():
            rows = ", ".join(f"{h.doc_id}:{h.score:.3f}" for h in out.hits)
            lines.append(f"{name} -> [{rows}]")
        lines.append("=====Actions=====")
        for a in self.log.actions:
            lines.append(f"node={a.node} server={a.server} op={a.op} detail={a.detail}")
        return "\n\n\n\n".join(lines)

    def _gather_doc_ids(self, node_names: Sequence[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for nm in node_names:
            out = self._outputs.get(nm)
            if not out:
                continue
            for h in out.hits:
                if h.doc_id not in seen:
                    seen.add(h.doc_id)
                    ordered.append(h.doc_id)
        return ordered

    def _exec_search(self, cfg: SearchNodeCfg) -> NodeOutput:
        if cfg.server not in self._servers:
            raise KeyError(f"server '{cfg.server}' not registered")
        srv = self._servers[cfg.server]

        if cfg.materialize_from:
            mat_ids = self._gather_doc_ids(cfg.materialize_from)
            if mat_ids:
                srv.ensure(mat_ids, from_nodes=cfg.materialize_from)
                self.log.actions.append(
                    ServerAction(
                        cfg.name,
                        srv.name,
                        "ensure",
                        {
                            "from_nodes": list(cfg.materialize_from),
                            "count": len(mat_ids),
                        },
                    )
                )

        candidates: Iterable[str] | None = None
        if cfg.select_from:
            candidates = self._gather_doc_ids(cfg.select_from)

        hits = srv.search(cfg.query, candidates=candidates, topk=cfg.topk)
        self.log.actions.append(
            ServerAction(
                cfg.name,
                srv.name,
                "search",
                {
                    "query": cfg.query,
                    "topk": cfg.topk,
                    "candidates": len(list(candidates)) if candidates is not None else None,
                },
            )
        )
        return NodeOutput(cfg.name, hits)

    def _exec_merge(self, cfg: MergeNodeCfg) -> NodeOutput:
        bucket: dict[str, list[float]] = {}
        for nm in cfg.sources:
            out = self._outputs.get(nm)
            if not out or not out.hits:
                continue
            scores = [h.score for h in out.hits]
            mn, mx = min(scores), max(scores)
            scale = (mx - mn) or 1.0
            for h in out.hits:
                s = (h.score - mn) / scale if cfg.score_norm else h.score
                bucket.setdefault(h.doc_id, []).append(s)

        merged: list[tuple[str, float]] = []
        if cfg.mode == "union":
            for doc, scs in bucket.items():
                merged.append((doc, (cfg.combiner(scs) if cfg.combiner else sum(scs) / len(scs))))
        elif cfg.mode == "intersect":
            for doc, scs in bucket.items():
                if len(scs) == len(cfg.sources):
                    merged.append((doc, (cfg.combiner(scs) if cfg.combiner else sum(scs) / len(scs))))
        else:
            raise ValueError("mode must be 'union' or 'intersect'")

        merged.sort(key=lambda x: x[1], reverse=True)
        if cfg.topk is not None:
            merged = merged[: cfg.topk]
        hits = [SearchHit(doc, sc) for doc, sc in merged]
        return NodeOutput(cfg.name, hits)

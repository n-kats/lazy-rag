from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, NotRequired, Protocol, TypedDict, cast, runtime_checkable

# ------- Config schemas -------


class ServerConfig(TypedDict):
    type: str
    name: str
    # 任意の追加フィールドは Any で受ける（各実装が読む）
    # ここでは mypy 的に許すために NotRequired[Any] を使う
    # 例: index_path など
    # Pydantic 側の model_dump() はこの形に準拠
    # why not: サーバごとの追加項目はレジストリ先の cls が解釈すれば十分
    extra: NotRequired[Any]


class SearchNodeConfig(TypedDict):
    type: Literal["search"]
    name: str
    server: ServerConfig
    query: str
    topk: NotRequired[int]
    from_nodes: NotRequired[list[str]]


class WorkflowConfig(TypedDict):
    type: Literal["workflow"]
    nodes: list[SearchNodeConfig]


# ------- Core types -------


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
    @property
    def type(self) -> str: ...

    def ensure(self, doc_ids: Iterable[str], *, from_nodes: Sequence[str]) -> None: ...
    def search(self, query: str, *, candidates: Iterable[str] | None = None, topk: int = 10) -> list[SearchHit]: ...

    def model_dump(self) -> ServerConfig: ...
    @classmethod
    def load_from_config(cls, config: ServerConfig) -> SearchServer: ...


@dataclass(frozen=True, slots=True)
class NodeOutput:
    name: str
    hits: list[SearchHit]


@dataclass(slots=True)
class SearchNodeCfg:
    type: Literal["search"] = "search"
    name: str = ""
    server: SearchServer | None = None
    query: str = ""
    topk: int = 10
    from_nodes: list[str] = field(default_factory=list)

    def model_dump(self) -> SearchNodeConfig:
        if self.server is None:
            raise ValueError("server is None in node dump")
        return {
            "type": "search",
            "name": self.name,
            "server": self.server.model_dump(),
            "query": self.query,
            "topk": self.topk,
            "from_nodes": list(self.from_nodes),
        }

    @classmethod
    def load_from_config(cls, config: SearchNodeConfig) -> SearchNodeCfg:
        srv_conf = config["server"]
        srv_type = srv_conf.get("type")
        if not isinstance(srv_type, str):
            raise TypeError("server.type must be str")
        server_cls = ServerRegistry.get(srv_type)
        server = server_cls.load_from_config(srv_conf)

        topk = config.get("topk", 10)
        if not isinstance(topk, int):
            raise TypeError("topk must be int")

        from_nodes = config.get("from_nodes", [])
        if not isinstance(from_nodes, list) or not all(isinstance(x, str) for x in from_nodes):
            raise TypeError("from_nodes must be list[str]")

        name = config["name"]
        query = config["query"]
        if not isinstance(name, str) or not isinstance(query, str):
            raise TypeError("name/query must be str")

        return cls(
            name=name,
            server=server,
            query=query,
            topk=topk,
            from_nodes=from_nodes,
        )


NodeCfg = SearchNodeCfg


# ------- Registry -------


class ServerRegistry:
    _reg: dict[str, type[SearchServer]] = {}

    @classmethod
    def register(cls, server_type: str, server_class: type[SearchServer]) -> None:
        cls._reg[server_type] = server_class

    @classmethod
    def get(cls, server_type: str) -> type[SearchServer]:
        try:
            return cls._reg[server_type]
        except KeyError as e:
            raise KeyError(f"unregistered server type: {server_type}") from e


# ------- Workflow -------


class Workflow:
    def __init__(self) -> None:
        self._nodes: list[NodeCfg] = []
        self._outputs: dict[str, NodeOutput] = {}
        self.log = ServerLog()

    def add(
        self,
        server: SearchServer,
        /,
        *,
        node_name: str | None = None,
        query: str,
        topk: int = 10,
        from_nodes: Sequence[str] = (),
    ) -> None:
        name = node_name or server.name
        cfg = SearchNodeCfg(
            name=name,
            server=server,
            query=query,
            topk=topk,
            from_nodes=list(from_nodes),
        )
        self._nodes.append(cfg)

    def run(self) -> dict[str, NodeOutput]:
        self._outputs.clear()
        for cfg in self._nodes:
            if cfg.type == "search":
                out = self._exec_search(cfg)
            else:
                raise ValueError(cfg.type)
            self._outputs[out.name] = out
        return dict(self._outputs)

    def report(self) -> str:
        lines: list[str] = ["===== Node Outputs ====="]
        for name, out in self._outputs.items():
            rows = ", ".join(f"{h.doc_id}:{h.score:.3f}" for h in out.hits)
            lines.append(f"{name} -> [{rows}]")
        lines.append("\n===== Actions =====")
        for a in self.log.actions:
            lines.append(f"node={a.node} server={a.server} op={a.op} detail={a.detail}")
        return "\n".join(lines)

    def model_dump(self) -> WorkflowConfig:
        return {
            "type": "workflow",
            "nodes": [n.model_dump() for n in self._nodes],
        }

    @classmethod
    def load_from_config(cls, config: WorkflowConfig) -> Workflow:
        if config.get("type") != "workflow":
            raise TypeError("config.type must be 'workflow'")

        nodes = config.get("nodes", [])
        if not isinstance(nodes, list):
            raise TypeError("nodes must be list")

        wf = cls()
        for nd in nodes:
            if not isinstance(nd, dict):
                raise TypeError("node must be dict")
            # TypedDict narrows with cast after runtime check
            node_cfg = SearchNodeCfg.load_from_config(cast(SearchNodeConfig, nd))
            wf._nodes.append(node_cfg)
        return wf

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
        if cfg.server is None:
            raise ValueError("server is None")
        srv = cfg.server

        candidates_list: list[str] | None = None
        if cfg.from_nodes:
            mat_ids = self._gather_doc_ids(cfg.from_nodes)
            if mat_ids:
                srv.ensure(mat_ids, from_nodes=cfg.from_nodes)
                self.log.actions.append(
                    ServerAction(
                        cfg.name, srv.name, "ensure", {"from_nodes": list(cfg.from_nodes), "count": len(mat_ids)}
                    )
                )
                candidates_list = mat_ids

        hits = srv.search(cfg.query, candidates=candidates_list, topk=cfg.topk)
        self.log.actions.append(
            ServerAction(
                cfg.name,
                srv.name,
                "search",
                {
                    "query": cfg.query,
                    "topk": cfg.topk,
                    "candidates": (len(candidates_list) if candidates_list is not None else None),
                },
            )
        )
        return NodeOutput(cfg.name, hits)

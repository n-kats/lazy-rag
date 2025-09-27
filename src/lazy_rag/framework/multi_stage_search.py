from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, NotRequired, Protocol, TypedDict, cast, runtime_checkable


class ServerConfig(TypedDict):
    type: str
    name: str
    extra: NotRequired[Any]


class SearchNodeConfig(TypedDict):
    name: str
    server_name: str
    topk: NotRequired[int]
    from_nodes: NotRequired[list[str]]


class WorkflowConfig(TypedDict):
    servers: list[ServerConfig]
    nodes: list[SearchNodeConfig]


@dataclass(frozen=True, slots=True)
class SearchHit:
    doc_id: str
    score: float
    payload: dict[str, object] = field(default_factory=dict)  # サーバー固有の情報


class Entry(TypedDict):
    source_server: str
    source_id: str
    payload: dict[str, object]  # サーバー毎に異なる詳細を保持


@dataclass(slots=True)
class ServerAction:
    node: str
    server_name: str
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

    def add_entries(self, entries: Sequence[Entry]) -> None: ...

    def search(
        self,
        query: str,
        topk: int,
    ) -> list[SearchHit]: ...

    def model_dump(self) -> ServerConfig: ...

    @classmethod
    def load_from_config(cls, config: ServerConfig) -> SearchServer: ...


@dataclass(frozen=True, slots=True)
class NodeOutput:
    node_name: str
    server_name: str
    hits: list[SearchHit]


@dataclass(slots=True)
class SearchNode:
    name: str
    server_name: str
    topk: int = 10
    from_nodes: list[str] = field(default_factory=list)

    def model_dump(self) -> SearchNodeConfig:
        return {
            "name": self.name,
            "server_name": self.server_name,
            "topk": self.topk,
            "from_nodes": list(self.from_nodes),
        }

    @classmethod
    def load_from_config(cls, config: SearchNodeConfig) -> SearchNode:
        topk = config.get("topk", 10)
        if not isinstance(topk, int):
            raise TypeError("topk must be int")

        from_nodes = config.get("from_nodes", [])
        if not isinstance(from_nodes, list) or not all(isinstance(x, str) for x in from_nodes):
            raise TypeError("from_nodes must be list[str]")

        name = config["name"]
        server_name = config["server_name"]
        if not isinstance(name, str) or not isinstance(server_name, str):
            raise TypeError("name/server_name must be str")
        return cls(
            name=name,
            server_name=server_name,
            topk=topk,
            from_nodes=from_nodes.copy(),
        )


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


class Workflow:
    def __init__(self) -> None:
        self._servers: dict[str, SearchServer] = {}
        self._nodes: list[SearchNode] = []

    def add(
        self,
        server: SearchServer,
        /,
        *,
        node_name: str,
        from_nodes: Sequence[str],
        topk: int,
    ) -> None:
        assert isinstance(server, SearchServer), "server must implement SearchServer"
        if server.name in self._servers:
            if self._servers[server.name] is not server:
                raise ValueError(f"duplicate server name: {server.name}")

        self._servers[server.name] = server
        name = node_name or server.name
        node = SearchNode(
            name=name,
            server_name=server.name,
            topk=topk,
            from_nodes=list(from_nodes),
        )
        assert all(n.name != name for n in self._nodes), f"duplicate node name: {name}"
        self._nodes.append(node)

    def search(self, query: Any) -> tuple[NodeOutput, list[NodeOutput], ServerLog]:
        assert self._nodes, "no nodes in workflow"
        trace = list[NodeOutput]()
        log = ServerLog()
        out: NodeOutput
        for node in self._nodes:
            out = self._exec_search(node, query, trace, log)
            trace.append(out)
        return out, trace, log

    def model_dump(self) -> WorkflowConfig:
        return {
            "servers": [s.model_dump() for s in self._servers.values()],
            "nodes": [n.model_dump() for n in self._nodes],
        }

    @classmethod
    def load_from_config(cls, config: WorkflowConfig) -> Workflow:
        node_configs = config.get("nodes", [])
        server_configs = config.get("servers", [])
        if not isinstance(node_configs, list) or not isinstance(server_configs, list):
            raise TypeError("nodes/servers must be list")

        servers_dict: dict[str, SearchServer] = {}
        for server_cfg in server_configs:
            if not isinstance(server_cfg, dict):
                raise TypeError("server must be dict")
            server = ServerRegistry.get(server_cfg["type"]).load_from_config(cast(ServerConfig, server_cfg))
            assert server.name not in servers_dict, f"duplicate server name: {server.name}"
            servers_dict[server.name] = server
        workflow = cls()
        for node_config in node_configs:
            if not isinstance(node_config, dict):
                raise TypeError("node must be dict")
            node = SearchNode.load_from_config(cast(SearchNodeConfig, node_config))
            workflow.add(
                servers_dict[node.server_name],
                node_name=node.name,
                topk=node.topk,
                from_nodes=node.from_nodes,
            )
        return workflow

    def _gather_entries_from_previous_nodes(self, trace: list[NodeOutput], node_names: Sequence[str]) -> list[Entry]:
        found_entries = list[Entry]()
        name_to_trace_node = {n.node_name: n for n in trace}
        for node_name in node_names:
            previous_node = name_to_trace_node[node_name]
            for hit in previous_node.hits:
                found_entries.append(
                    Entry(
                        source_server=previous_node.server_name,
                        source_id=hit.doc_id,
                        payload=hit.payload,
                    )
                )
        return found_entries

    def _exec_search(self, node: SearchNode, query: Any, trace: list[NodeOutput], log: ServerLog) -> NodeOutput:
        server = self._servers[node.server_name]
        entries_from_previous_nodes = self._gather_entries_from_previous_nodes(trace, node.from_nodes)

        if entries_from_previous_nodes:
            server.add_entries(entries_from_previous_nodes)
            log.actions.append(
                ServerAction(
                    node.name,
                    server.name,
                    "add_entries",
                    {
                        "from_nodes": list(node.from_nodes),
                        "count": len(entries_from_previous_nodes),
                    },
                )
            )

        hits = server.search(query, topk=node.topk)
        log.actions.append(
            ServerAction(
                node.name,
                server.name,
                "search",
                {
                    "query": query,
                    "topk": node.topk,
                },
            )
        )
        return NodeOutput(
            node_name=node.name,
            server_name=server.name,
            hits=hits,
        )


def show_trace(trace: list[NodeOutput], log: ServerLog) -> str:
    lines: list[str] = ["===== Node Outputs ====="]
    for node_output in trace:
        name = node_output.node_name
        rows = ", ".join(f"{hit.doc_id}:{hit.score:.3f}" for hit in node_output.hits)
        lines.append(f"{name} -> [{rows}]")
    lines.append("\n===== Actions =====")
    for action in log.actions:
        lines.append(f"node={action.node} server={action.server_name} op={action.op} detail={action.detail}")
    return "\n".join(lines)

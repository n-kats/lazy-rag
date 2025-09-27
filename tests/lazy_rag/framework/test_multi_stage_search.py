import pytest
from typing import Any
from lazy_rag.framework.multi_stage_search import (
    SearchHit,
    Entry,
    SearchNode,
    SearchNodeConfig,
    ServerConfig,
    ServerRegistry,
    Workflow,
    show_trace,
    SearchServer,
    NodeOutput,
    ServerAction,
    ServerLog,
)


class DummyServer:
    """簡易モック SearchServer 実装"""

    def __init__(self, name: str, stype: str):
        self._name = name
        self._type = stype
        self._entries: list[Entry] = []
        self._queries: list[tuple[str, int]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> str:
        return self._type

    def add_entries(self, entries: list[Entry]) -> None:
        self._entries.extend(entries)

    def search(self, query: str, topk: int) -> list[SearchHit]:
        self._queries.append((query, topk))
        # 疑似ヒットを返す
        return [SearchHit(doc_id=f"{self._name}-{i}", score=1.0 - i*0.1) for i in range(topk)]

    def model_dump(self) -> ServerConfig:
        return {"type": self.type, "name": self.name}

    @classmethod
    def load_from_config(cls, config: ServerConfig) -> "DummyServer":
        return cls(name=config["name"], stype=config["type"])


def test_searchnode_load_from_config_valid():
    cfg: SearchNodeConfig = {
        "type": "search",
        "name": "n1",
        "server_name": "s1",
        "topk": 5,
        "from_nodes": ["prev"],
    }
    node = SearchNode.load_from_config(cfg)
    assert node.name == "n1"
    assert node.topk == 5
    assert node.from_nodes == ["prev"]


def test_server_registry_register_and_get():
    ServerRegistry.register("dummy", DummyServer)
    cls = ServerRegistry.get("dummy")
    assert cls is DummyServer


def test_server_registry_unregistered():
    with pytest.raises(KeyError):
        ServerRegistry.get("unknown")


def test_workflow_add_and_search(monkeypatch):
    server = DummyServer("s1", "dummy")
    wf = Workflow()
    wf.add(server, node_name="n1", from_nodes=[], topk=2)

    # 実行
    out, trace, log = wf.search("query")
    assert isinstance(out, NodeOutput)
    assert out.node_name == "n1"
    assert out.hits  # ダミーヒットが返ってくる

    # show_trace 出力確認
    text = show_trace(trace, log)
    assert "===== Node Outputs =====" in text


def test_exec_search_with_from_nodes():
    server1 = DummyServer("s1", "dummy")
    server2 = DummyServer("s2", "dummy")

    wf = Workflow()
    wf._servers = {"s1": server1, "s2": server2}
    node1 = SearchNode("n1", "s1", topk=1)
    node2 = SearchNode("n2", "s2", topk=1, from_nodes=["n1"])
    wf._nodes = [node1, node2]

    trace: list[NodeOutput] = []
    log = ServerLog()

    # 1段目
    out1 = wf._exec_search(node1, "hello", trace, log)
    trace.append(out1)

    # 2段目（from_nodes 経由で add_entries 呼ばれる）
    out2 = wf._exec_search(node2, "world", trace, log)
    assert server2._entries  # エントリ追加されている
    assert any(a.op == "add_entries" for a in log.actions)
    assert any(a.op == "search" for a in log.actions)

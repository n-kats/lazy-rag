from lazy_rag.framework.multi_stage_search import Workflow, SearchNode, SearchHit, show_trace, ServerRegistry, ServerConfig, Entry
from typing import Sequence


# 簡易なダミーサーバー
class DummyServer:
    def __init__(self, name: str, stype: str):
        self._name = name
        self._type = stype
        self._entries: list[Entry] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> str:
        return self._type

    def add_entries(self, entries: Sequence[Entry]) -> None:
        print(
            f"[DummyServer:{self.name}] add_entries called with {len(entries)} entries")
        self._entries.extend(entries)

    def search(self, query: str, topk: int):
        print(
            f"[DummyServer:{self.name}] search called with query={query!r}, topk={topk}")
        return [
            SearchHit(doc_id=f"{self.name}-doc{i}", score=1.0 - 0.1 * i)
            for i in range(topk)
        ]

    def model_dump(self) -> ServerConfig:
        return {"type": self.type, "name": self.name}

    @classmethod
    def load_from_config(cls, config: ServerConfig) -> "DummyServer":
        return cls(config["name"], config["type"])


# ---- サンプル実行 ----
if __name__ == "__main__":
    # レジストリに登録
    ServerRegistry.register("dummy", DummyServer)

    # サーバー用意
    s1 = DummyServer("server1", "dummy")
    s2 = DummyServer("server2", "dummy")

    # ワークフロー構築
    wf = Workflow()
    wf.add(s1, node_name="n1", from_nodes=[], topk=2)
    wf.add(s2, node_name="n2", from_nodes=["n1"], topk=3)

    # 検索実行
    query = "example query"
    final_out, trace, log = wf.search(query)

    # 出力確認
    print("\n=== show_trace output ===")
    print(show_trace(trace, log))

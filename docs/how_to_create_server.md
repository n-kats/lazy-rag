# 検索サーバー実装ガイド

LazyRAG の検索サーバーは `SearchServer` プロトコルに準拠する必要があります。

## 必須インターフェース
```
    class SearchServer(Protocol):
        @property
        def name(self) -> str: ...
        @property
        def type(self) -> str: ...

        def ensure(self, doc_ids: Iterable[str], *, from_nodes: Sequence[str]) -> None: ...
        def search(
            self,
            query: str,
            *,
            candidates: Iterable[str] | None = None,
            topk: int = 10,
        ) -> list[SearchHit]: ...

        def model_dump(self) -> dict[str, object]: ...
        @classmethod
        def load_from_config(cls, config: dict[str, object]) -> SearchServer: ...
```
- **name**: ワークフロー内の一意な識別子  
- **type**: サーバー種別（例: `"bm25"`, `"embedding"`）  
- **ensure**: ドキュメントを準備（インデックス登録など、副作用のみ）  
- **search**: クエリ検索、`SearchHit` のリストを返す  
- **model_dump / load_from_config**: 設定ファイルへの保存・復元用  

## 最小実装例

    class BM25Server:
        def __init__(self, name: str, index_path: str) -> None:
            self._name = name
            self._index_path = index_path

        @property
        def name(self) -> str: return self._name
        @property
        def type(self) -> str: return "bm25"

        def ensure(self, doc_ids, *, from_nodes): pass
        def search(self, query, *, candidates=None, topk=10): return []

        def model_dump(self) -> dict[str, object]:
            return {"type": self.type, "name": self.name, "index_path": self._index_path}

        @classmethod
        def load_from_config(cls, config: dict[str, object]) -> SearchServer:
            return cls(name=str(config["name"]), index_path=str(config["index_path"]))

登録:

    ServerRegistry.register("bm25", BM25Server)

## チェックリスト

- [ ] `ensure` は副作用のみ、返り値なし  
- [ ] `search` は純粋関数的に動作  
- [ ] `model_dump` / `load_from_config` で round-trip 可能  
- [ ] `ServerRegistry.register()` 済み
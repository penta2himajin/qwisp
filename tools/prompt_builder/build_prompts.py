#!/usr/bin/env python3
"""Qwisp — ベンチ由来プロンプト builder（決定論的）。

選定ベンチ（docs/04-benchmark-selection.md）のサブセットを取得して、Step 1 collector の
プロンプトスキーマ（{prompt_id, category, text}）に正規化し `prompts.jsonl` を生成する。

設計:
- **stdlib のみ**（urllib/json/gzip/random/zlib）。datasets ライブラリ不要 → loading-script /
  trust_remote_code の非決定性を回避。MTPLX runtime-venv も汚さない。
- フェッチは HF の生ファイル（`/resolve/main/`）と datasets-server `/rows` REST。
  公開データなので token 不要。`HF_TOKEN` が env にあればレート制限緩和に使う（任意）。
- **決定論**: seed 固定。各ベンチで pool を作り seeded sample。選択 id を lock に記録。
- 比率 45/45/10 は manifest の `n` で担保。

使い方:
    python build_prompts.py --manifest benchmarks.manifest.json \
        --out ../step1_routing_trace/prompts.jsonl --lock prompts.lock.json
    python build_prompts.py ... --limit-per-bench 2   # 検証用に各ベンチ n を上書き
"""

import argparse
import gzip
import hashlib
import json
import os
import random
import sys
import tempfile
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor

UA = {"User-Agent": "qwisp-prompt-builder/0.1"}
DS_SERVER = "https://datasets-server.huggingface.co"
HF = "https://huggingface.co"


def _req(url, rng=None):
    h = dict(UA)
    tok = os.environ.get("HF_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    if rng:
        h["Range"] = f"bytes=0-{rng}"
    return urllib.request.Request(url, headers=h)


def http_bytes(url, rng=None, timeout=90):
    last = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(_req(url, rng), timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001 - 簡易リトライ
            last = e
    raise last


def http_json(url, timeout=90):
    return json.loads(http_bytes(url, timeout=timeout))


CACHE_DIR = os.path.join(tempfile.gettempdir(), "qwisp_build_cache")


def http_bytes_cached(url, rng=None, timeout=120):
    """巨大DL（lcb の Range, repoqa gz）用のオンディスクキャッシュ。再実行が即時化。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = hashlib.sha256(f"{url}#r{rng}".encode()).hexdigest()[:16]
    path = os.path.join(CACHE_DIR, key)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    data = http_bytes(url, rng=rng, timeout=timeout)
    with open(path, "wb") as f:
        f.write(data)
    return data


def pmap(fn, items, workers=8):
    """順序保存の並行 map（I/O bound のフェッチを並行化）。"""
    items = list(items)
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as ex:
        return list(ex.map(fn, items))


def seeded(seed, name):
    """ベンチ名で安定に派生した RNG（hash() は randomized なので zlib.crc32 を使う）。"""
    return random.Random(seed ^ zlib.crc32(name.encode()))


# ---- datasets-server /rows ヘルパ ----

def ds_count(dataset, config, split):
    u = f"{DS_SERVER}/rows?dataset={dataset}&config={config}&split={split}&offset=0&length=1"
    return http_json(u)["num_rows_total"]


def ds_row(dataset, config, split, offset):
    u = f"{DS_SERVER}/rows?dataset={dataset}&config={config}&split={split}&offset={offset}&length=1"
    return http_json(u)["rows"][0]["row"]


def hf_raw(repo, path, rng=None):
    return http_bytes(f"{HF}/datasets/{repo}/resolve/main/{path}", rng=rng)


def hf_tree(repo, path=""):
    return http_json(f"{HF}/api/datasets/{repo}/tree/main/{path}".rstrip("/"))


def jsonl_lines(blob):
    for line in blob.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


# ---- アダプタ（各々 (prompts, lock_ids) を返す）----

def adapt_swe_verified(src, n, seed):
    ds, cfg, sp = src["dataset"], src.get("config", "default"), src.get("split", "test")
    total = ds_count(ds, cfg, sp)
    idxs = sorted(seeded(seed, "swe").sample(range(total), min(n, total)))
    rows = pmap(lambda i: ds_row(ds, cfg, sp, i), idxs)  # 並行フェッチ（順序保存）
    prompts, ids = [], []
    for row in rows:
        pid = row["instance_id"]
        text = (f"Repository: {row['repo']} @ {row['base_commit'][:10]}\n"
                f"Resolve the following GitHub issue.\n\n{row['problem_statement']}")
        prompts.append({"prompt_id": f"swe/{pid}", "category": "coding", "text": text})
        ids.append(pid)
    return prompts, ids


def adapt_livecodebench(src, n, seed):
    # test.jsonl は private_test_cases 込みで巨大。9件 sample のため全DLは無駄なので
    # Range で先頭 max_bytes だけ取得（最後の不完全行は捨てる）。キャッシュで再実行即時化。
    max_bytes = src.get("max_bytes", 12_000_000)
    url = f"{HF}/datasets/{src['repo']}/resolve/main/{src.get('file', 'test.jsonl')}"
    blob = http_bytes_cached(url, rng=max_bytes - 1)
    text = blob.decode("utf-8", "replace")
    lines = text.split("\n")
    if not text.endswith("\n"):
        lines = lines[:-1]  # Range で切れた末尾の不完全行を捨てる
    pool = [json.loads(ln) for ln in lines if ln.strip()]
    picks = seeded(seed, "lcb").sample(pool, min(n, len(pool)))
    prompts, ids = [], []
    for r in picks:
        qid = str(r.get("question_id", r.get("question_title", "?")))
        text = r["question_content"]
        if r.get("starter_code"):
            text += f"\n\nComplete the starter code:\n{r['starter_code']}"
        prompts.append({"prompt_id": f"lcb/{qid}", "category": "coding", "text": text})
        ids.append(qid)
    return prompts, ids


def adapt_bfcl(src, n, seed):
    # multi_turn 系は function を持たず別スキーマ（func は multi_turn_func_doc/ 側）なので、
    # function+question を持つ単一ターン系のみを pool にする。欠落は skip。
    pool = []
    for fn in src["files"]:
        for r in jsonl_lines(hf_raw(src["repo"], fn)):
            if "function" in r and "question" in r:
                pool.append(r)
    picks = seeded(seed, "bfcl").sample(pool, min(n, len(pool)))
    prompts, ids = [], []
    for r in picks:
        rid = r["id"]
        # question: [[{role,content}...]] （最初のターン群の user 発話を採用）
        q = r["question"]
        turns = q[0] if q and isinstance(q[0], list) else q
        user = next((t.get("content", "") for t in turns if t.get("role") == "user"), "")
        tools = json.dumps(r["function"], ensure_ascii=False)
        text = (f"You are a function-calling agent. Available tools:\n{tools}\n\n"
                f"User request: {user}\n\nRespond with the appropriate function call(s).")
        prompts.append({"prompt_id": f"bfcl/{rid}", "category": "agentic", "text": text})
        ids.append(rid)
    return prompts, ids


def adapt_terminal_bench(src, n, seed):
    tree = hf_tree(src["repo"])
    task_dirs = sorted(t["path"] for t in tree if t.get("type") == "directory")
    picks = sorted(seeded(seed, "tbench").sample(task_dirs, min(n, len(task_dirs))))

    def fetch(d):
        try:
            return d, hf_raw(src["repo"], f"{d}/instruction.md").decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            print(f"[builder] terminal-bench skip {d}: {e}", file=sys.stderr)
            return d, None

    prompts, ids = [], []
    for d, ins in pmap(fetch, picks):  # 並行フェッチ（順序保存）
        if ins is None:
            continue
        prompts.append({"prompt_id": f"tbench/{d}", "category": "agentic", "text": ins})
        ids.append(d)
    return prompts, ids


def adapt_ruler(src, n, seed):
    ds, sp = src["dataset"], src.get("split", "validation")
    configs = src["configs"]
    rng = seeded(seed, "ruler")
    prompts, ids = [], []
    for j in range(n):
        cfg = configs[j % len(configs)]
        total = ds_count(ds, cfg, sp)
        off = rng.randrange(total)
        row = ds_row(ds, cfg, sp, off)
        prompts.append({"prompt_id": f"ruler/{cfg}/{off}", "category": "long_context",
                        "text": row["input"]})
        ids.append(f"{cfg}/{off}")
    return prompts, ids


def adapt_repoqa(src, n, seed):
    v = src.get("version", "2024-06-23")
    budget = src.get("context_char_budget", 48000)
    url = (f"https://github.com/evalplus/repoqa_release/releases/download/"
           f"{v}/repoqa-{v}.json.gz")
    obj = json.loads(gzip.decompress(http_bytes_cached(url)))
    flat = []  # (lang, entry)
    for lang, entries in obj.items():
        for e in entries:
            flat.append((lang, e))
    picks = seeded(seed, "repoqa").sample(flat, min(n, len(flat)))
    prompts, ids = [], []
    for lang, e in picks:
        # content(dict path->code) を sorted で連結し budget 文字で打ち切り＝長文脈。
        parts, used = [], 0
        for path in sorted(e["content"]):
            code = e["content"][path]
            chunk = f"# ===== {path} =====\n{code}\n"
            if used + len(chunk) > budget:
                break
            parts.append(chunk)
            used += len(chunk)
        needle = e["needles"][0] if e.get("needles") else {}
        desc = needle.get("description") or needle.get("name", "the target function")
        text = ("Below is a long code context. Based on the function description, find the "
                "matching function and return its name.\n\n"
                f"Function description: {desc}\n\n=== CODE ({lang}: {e['repo']}) ===\n"
                + "".join(parts))
        rid = f"{e['repo']}/{needle.get('name', '?')}"
        prompts.append({"prompt_id": f"repoqa/{rid}", "category": "long_context", "text": text})
        ids.append(rid)
    return prompts, ids


ADAPTERS = {
    "swe_verified": adapt_swe_verified,
    "livecodebench": adapt_livecodebench,
    "bfcl": adapt_bfcl,
    "terminal_bench": adapt_terminal_bench,
    "ruler": adapt_ruler,
    "repoqa": adapt_repoqa,
}


def main():
    ap = argparse.ArgumentParser(description="Qwisp benchmark-derived prompt builder")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True, help="prompts.jsonl 出力先")
    ap.add_argument("--lock", required=True, help="再現用 lock JSON 出力先")
    ap.add_argument("--limit-per-bench", type=int, default=None, help="各ベンチ n を上書き（検証用）")
    ap.add_argument("--only", help="このベンチ名だけ実行（検証用）")
    ap.add_argument("--jobs", type=int, default=6, help="ベンチ間の並行数")
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    seed = manifest.get("seed", 0)

    benches = [b for b in manifest["benchmarks"] if not args.only or b["name"] == args.only]

    def run_one(bench):
        name = bench["name"]
        n = args.limit_per_bench or bench["n"]
        print(f"[builder] {name}: fetching n={n} ...", file=sys.stderr)
        try:
            prompts, ids = ADAPTERS[bench["adapter"]](bench["source"], n, seed)
            print(f"[builder] {name}: +{len(prompts)} ({bench['category']})", file=sys.stderr)
            return bench, prompts, ids
        except Exception as e:  # noqa: BLE001 - 1ベンチ失敗で全体を止めない
            print(f"[builder] {name}: FAIL {type(e).__name__}: {e}", file=sys.stderr)
            return bench, None, None

    # ベンチ間を並行実行。ex.map は入力順を保つので出力は manifest 順で決定論的。
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        results = list(ex.map(run_one, benches))

    all_prompts, lock = [], {"seed": seed, "selected": {}}
    counts = {}
    for bench, prompts, ids in results:
        if prompts is None:
            continue
        all_prompts.extend(prompts)
        lock["selected"][bench["name"]] = ids
        counts[bench["category"]] = counts.get(bench["category"], 0) + len(prompts)

    with open(args.out, "w", encoding="utf-8") as f:
        for p in all_prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(args.lock, "w", encoding="utf-8") as f:
        json.dump(lock, f, ensure_ascii=False, indent=2)

    tot = sum(counts.values()) or 1
    ratio = {k: round(v / tot * 100) for k, v in counts.items()}
    print(f"[builder] wrote {len(all_prompts)} prompts -> {args.out}", file=sys.stderr)
    print(f"[builder] counts={counts} ratio={ratio}", file=sys.stderr)
    print(f"[builder] lock -> {args.lock}", file=sys.stderr)


if __name__ == "__main__":
    main()

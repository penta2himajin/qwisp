# corpora/ — 長文脈10%枠の実素材

`prompts.jsonl` の `long_context` 行が `text_file` で参照する大きな実素材を置く場所。
**用途でなく「キャッシュ膨張のストレス検査」枠**なので、短文の合成ではなく
**実運用に近い 8K–16K tokens の本物**を入れること（同ドメイン＝長いコード／長いエージェント履歴）。

参照ファイル（`prompts.jsonl` 既定）:
- `long_code_dump.txt` — 長い実コード（複数ファイル連結 or 大きめのソース1本）。
  例: `cat ~/repos/<your-repo>/src/*.py > long_code_dump.txt`
- `long_agent_history.txt` — 長いエージェント tool-calling 履歴（実ログを貼る）。

このディレクトリの中身は `.gitignore` 済み（各自の素材なので未追跡）。
coding/agentic の18行は素材不要で即実行できる。long_context 2行はここを埋めてから回す。

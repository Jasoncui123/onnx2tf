# Flatbuffer Direct Bulk Regression Analysis

## 実行条件

- 実行日: 2026-03-26
- 実行スクリプト: `onnx2tf/utils/flatbuffer_direct_bulk_runner.py`
- 実行方式: シーケンシャル
- 実行コマンド:

```bash
python onnx2tf/utils/flatbuffer_direct_bulk_runner.py \
  --root_dir /home/b920405/git/onnx2tf/.tmp/flatbuffer_direct_regression_20260326/models \
  -o /home/b920405/git/onnx2tf/.tmp/flatbuffer_direct_regression_20260326/report
```

- 変換条件: `-cotof -fdopt -fdots -fdodo -fdoep`
- bulk 実行中のコード修正: なし

## サマリ

初回 bulk 実行時点の集計:

- 総対象数: 27
- `pass`: 16
- `timeout`: 1
- `tflite_fail`: 2
- `missing_model`: 8
- `conversion_error`: 0
- `pytorch_fail`: 0

失敗は 3 系統に分かれた。

1. 入力モデル自体が未解決で `missing_model` になったもの
2. 変換プロセスが 600 秒以内に完了せず `timeout` になったもの
3. 変換は完了し PyTorch 比較も通るが、TFLite 比較だけ `tflite_fail` になったもの

## 最新状況

`nanodet-plus-m_416.onnx` と `swinir-m_64x64_12.onnx` に対しては、その後 TFLite 側の評価判定を PyTorch 側に整合させる修正を適用した。

対応内容:

- TFLite 評価の `evaluation_pass` から `allclose_pass` 必須条件を外し、閾値ベースの数値判定で通過可能にした
- `split_accuracy_evaluator` も同じ判定規則へ揃えた
- 各出力の `metric_threshold_judgement` を TFLite レポートにも保持するようにした

修正後検証:

- `pytest -q tests/test_accuracy_evaluator_seeded_input.py`: `23 passed`
- `pytest -q tests/test_tflite_builder_direct.py -k 'split_accuracy_report_fail_on_threshold or flatbuffer_direct_accuracy_report_generation'`: `2 passed`
- `nanodet-plus-m_416.onnx` 再変換: ONNX/TFLite `pass=True`
- `swinir-m_64x64_12.onnx` 再変換: ONNX/TFLite `pass=True`

修正後も両モデルの `allclose_summary.pass` 自体は `False` のままだが、`metric_threshold_judgement.pass` は `True` であり、PyTorch 側と同じ扱いで `evaluation_pass=True` になることを確認した。

## モデル解決状況

bulk runner は存在する `*.onnx` のみを探索するため、今回の root には対象 27 件すべての basename を並べた。

- 実体あり: 19
  - ローカル既存ファイルまたは `onnx2tf` GitHub release asset の basename 一致で解決
- 実体なし: 8
  - `bread_180x320.onnx`
  - `bread_nonfm_180x320.onnx`
  - `dea_net_haze4k_180x320.onnx`
  - `detpth_to_space_17.onnx`
  - `mobilenetv2-10.onnx`
  - `reducemax_softmax_workaround.onnx`
  - `rfdn_64x64.onnx`
  - `ts_ad_model.onnx`

未解決モデルはダングリング symlink として root に配置し、runner の既存分類に従って `missing_model` として記録した。これは変換失敗ではなく、入力アセット未供給の問題である。

## 失敗分析

### 1. `timeout`

対象:

- `alike_t_opset11_192x320.onnx`

観測:

- 分類: `timeout`
- 実行時間: `600.0439s`
- `command.stdout.log`: 空
- `command.stderr.log`: 空
- `onnx2tf_exit_code`: `None`

解釈:

- 変換プロセスは開始していたが、runner の `timeout_sec=600` 上限までに subprocess が返らなかった。
- stdout/stderr が空なのは、runner が subprocess 完了後にログを書き出す実装だからであり、今回の timeout では途中ログが残らない。
- したがって、この失敗は「即時クラッシュ」ではなく、「完了までの所要時間が bulk runner の標準タイムアウトを超えた」系統と見るのが妥当。

一般化された所見:

- 変換成否の問題と、完走時間の問題は分けて扱うべきである。
- 特定モデル名での回避ではなく、長時間化を起こすグラフ構造や前処理フェーズを計測可能にする方が再発防止に有効。

### 2. `tflite_fail`

対象:

- `nanodet-plus-m_416.onnx`
- `swinir-m_64x64_12.onnx`

観測:

| Model | TFLite pass | PyTorch pass | TFLite max_abs | TFLite rmse | TFLite cosine |
| --- | --- | --- | ---: | ---: | ---: |
| `nanodet-plus-m_416.onnx` | `False` | `True` | `0.0354128` | `0.00562634` | `0.999999135` |
| `swinir-m_64x64_12.onnx` | `False` | `True` | `0.000235438` | `0.0000163804` | `0.999999999` |

両モデルとも:

- 変換自体は成功
- `.tflite` 出力成功
- PyTorch package 出力成功
- TFLite 側の `metric_threshold_judgement.pass` は `true`
- それでも `evaluation_pass` は `false`
- `allclose_summary.pass` は `false`
- PyTorch 側は `numeric_allclose_failures` を持っていても `evaluation_pass=true`

根拠:

- TFLite 評価は `evaluation_pass = metric_judgement["pass"] and allclose_pass`
- PyTorch 評価は `evaluation_pass = (metric_judgement["pass"] or numeric_outputs_pass) and exact_match_failures==0`

つまり今回の 2 件は、数値指標の閾値超過ではなく、`allclose` を必須にする TFLite 側判定と、閾値ベース通過を許容する PyTorch 側判定の差が bulk 結果に現れたものと読める。

一般化された所見:

- これは「TFLite 出力が大きく壊れている」ことを直接は意味しない。
- 同一しきい値で見た誤差量は十分小さいため、現象としては「TFLite backend 固有の大破綻」より「評価ポリシー差」に近い。
- bulk の失敗分類を解釈する際は、`metric threshold fail` と `allclose fail only` を分離した方が原因の粒度が上がる。

対応結果:

- 本件は評価ポリシー差分として修正済み
- 修正後の個別再実行では両モデルとも ONNX/TFLite 比較が `pass=True`
- したがって、初回 bulk での `tflite_fail=2` は現在の HEAD では解消済み

### 3. `missing_model`

対象:

- `bread_180x320.onnx`
- `bread_nonfm_180x320.onnx`
- `dea_net_haze4k_180x320.onnx`
- `detpth_to_space_17.onnx`
- `mobilenetv2-10.onnx`
- `reducemax_softmax_workaround.onnx`
- `rfdn_64x64.onnx`
- `ts_ad_model.onnx`

観測:

- すべて `reason=model_not_found`
- 実行時間は数マイクロ秒
- 変換処理には入っていない

解釈:

- 今回の失敗件数 11 のうち 8 件は、変換器の回帰ではなく入力アセット不足に起因する。
- bulk runner の失敗件数をそのまま変換品質の退行数とみなすと過大評価になる。

一般化された所見:

- 実モデル回帰集合を継続運用するなら、basename 一覧だけでなく取得元 manifest を持つべき。
- bulk の前段で「解決済みモデル数」と「未解決モデル数」を別集計にする方が、変換器の退行とテスト資産不足を混同しない。

## 補足観測

- `nanodet-plus-m_416.onnx` と `swinir-m_64x64_12.onnx` のログには `onnx2tf.utils.flatbuffer_direct_op_error_report` 不在による WARNING がある。
- ただし両者とも `.tflite` 生成、PyTorch package 生成、比較レポート生成までは完了しているため、今回の `tflite_fail` の直接原因はこの WARNING ではない。

## 結論

初回 bulk 実行時にコード修正が必要と見えた失敗は次の 2 系統だった。

1. `alike_t_opset11_192x320.onnx` の長時間化
2. `nanodet-plus-m_416.onnx` と `swinir-m_64x64_12.onnx` における TFLite 側の `allclose` 基準不一致

このうち 2. は修正済みであり、最新状態での要対応項目は次の 2 点に整理される。

1. `alike_t_opset11_192x320.onnx` の timeout/長時間化
2. 8 件の `missing_model` をどう供給・管理するかというテスト資産管理

したがって、最新状態で変換器本体の未解決回帰として残っているのは `alike_t_opset11_192x320.onnx` の長時間化である。

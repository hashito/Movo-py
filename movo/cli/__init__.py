"""movo のコマンドライン。

構成:

  ``main.py``       入口。引数を読み、コマンドへ振り分け、終了コードを決める
  ``args.py``       小さな引数パーサ（JS 版と «書き方» を揃えるため自前）
  ``help.py``       ヘルプ本文
  ``pipeline.py``   JSON → 検証 → 描画 → 書き出しの筋道
  ``parallel.py``   **並列レンダリング（速度の要）**
  ``bridge.py``     他の担当が移植中のモジュールへの繋ぎ口
  ``console.py``    ログ・色・進捗
  ``errors.py``     エラーと終了コード
  ``templates.py``  ``movo init`` の雛形
  ``commands/``     コマンドごとの実装

**ここで重いものを import しないでください。** `movo --version` が
NumPy と Numba を読み込むと 1 秒近くかかります。実際に使うコマンドの中で
読むようにしてあります。
"""

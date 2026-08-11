"""Movo — JSON から動画を作る。

速度の設計判断は README.ja.md に実測つきで書いてあります。要点だけ:

- 全画面の演算は **NumPy**（純 Python の 54 倍）
- 画素ごとのループは **Numba**（NumPy 一括判定の 103 倍）
- フレーム単位で **multiprocessing**
"""

__version__ = "0.1.0"

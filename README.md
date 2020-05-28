# transformerを実装した

dataディレクトリに指定のファイルを入れる

en_ja_8000.model
train50000jatext.txt
train50000entext.txt

embedding層と出力層は共有
日本語，英語の両方のtextを用いてsentencepiece_vocab8000,unigramのパラメータで学習




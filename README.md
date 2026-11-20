# gpt2-from-scratch

GPT-2（124M）の学習プログラムを、Andrej Karpathyの[llm.c](https://github.com/karpathy/llm.c)の`train_gpt2.c`を参考に、Cで一から書き直したものです。

## やったこと

- 順伝播と逆伝播を、層ごとに手で実装（`train_gpt2.c`、約1,200行）
- 1関数ずつコミットしながら実装
- 各処理が何をしているかを、日本語のコメントで説明

## 進め方

1. AIに全体の骨組みを作ってもらう
2. 各関数の中身を自分で考えて書く
3. 詰まったらAIにヒントをもらい、llm.cの実装に近づくまで書き直す

## 実行方法

学習済みの重み（`gpt2_124M.bin`）、トークナイザ（`gpt2_tokenizer.bin`）、学習データ（`dev/data/tinyshakespeare/`）は、llm.cのREADMEの手順で作成し、このディレクトリに置きます。

```sh
cc -O3 train_gpt2.c -lm -o train_gpt2
./train_gpt2
```

## 出典とライセンス

- このリポジトリは、Andrej Karpathyの[llm.c](https://github.com/karpathy/llm.c)をもとにしています
- `llmc/`以下のファイルは、llm.cのものをそのまま使っています
- llm.cはMIT Licenseで公開されています。ライセンス全文は[LICENSE](LICENSE)を参照してください

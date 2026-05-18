typedef struct {
    int max_seq_len; // 入力トークンの最大個数。
    int vocab_size; // 語彙の種類数。
    int padded_vocab_size; // vocab_sizeを128の倍数に切り上げた値、CUDAカーネル効率化のため。
    int num_layers; // Transformerのブロック数。
    int num_heads; // 並列で実行されるAttention関数の個数。
    int channels; // 各トークンのベクトルの要素数。
} GPT2Config;

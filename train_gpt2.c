typedef struct {
    int max_seq_len; // 入力トークンの最大個数。
    int vocab_size; // 語彙の種類数。
    int padded_vocab_size; // vocab_sizeを128の倍数に切り上げた値、CUDAカーネル効率化のため。
    int num_layers; // Transformerのブロック数。
    int num_heads; // 並列で実行されるAttention関数の個数。
    int channels; // 各トークンのベクトルの要素数。
} GPT2Config;

typedef struct {
    float* wte; // 埋め込み行列(V, C)、トークンIDをベクトルに変換する・logits計算前に次元をVに戻す役割を持つ
    float* wpe; // 位置ベクトル(maxT, C)。トークン列を並列で処理するため、順序を持たない。そのため、位置情報を加える。
    float* ln1w; // 一層目の正規化(V, C)。
    float* ln1b; // バイアス項(V, C)。
    float* qkvw; // 入力トークンからQKVを生成するための重み行列(T, 3*C, C)。
    float* qkvb; // バイアス項(T, 3*C)。
    float* attprojw; // h個のAttentionの出力を連結したものを重みづけする役割を持つ。(T, C)
    float* attprojb; // バイアス項(T, C)
    float* fcw; // FFN層の重み行列(T, 4*C)。次元を４倍に拡張し、表現力をあげる。
    float* fcb; // バイアス項(T, 4*C)
    float* fcprojw; // GeLU関数後に次元を元に戻す重み行列(T, C)。
    float* fcprojb; // バイアス項(T, C)。
    float* lnfw; // 最終層のLN（T, C)。加算され続けた分散のスケールを安定させる。
    float* lnfb; // バイアス項(T, C)。
} ParameterTensors;

void encoder_forward(float* out, int* inp, float* wte, float* wpe, int B, int T, int C) {
    // out: (B, T, C)
    // inp: (B, T)
    // wte: (V, C)
    // wpe: (maxT, C)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int ix = inp[b * T + t];
            float* out_bt = out + b * T * C + t * C;
            float* wte_ix = wte + ix * C;
            float* wpe_t = wpe + t * C;
            for (int c = 0; c < C; c++) {
                out_bt[c] = wte_ix[c] + wpe_t[c];
            }
        }
    }
}

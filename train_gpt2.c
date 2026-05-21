#include <math.h>

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

void layernorm_forward(float* out, float* mean, float* rstd,
                        float* inp, float* weight, float* bias,
                         int B, int T, int C) {
    // out, inp: (B, T, C)
    // weight, bias: (C)
    // mean, rstd: (B, T)
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* inp_bt = inp + b * T * C + t * C;
            float m = 0;
            for (int c = 0; c < C; c++) {
                m += inp_bt[c];
            }
            m /= C;
            float v = 0;
            for (int c = 0; c < C; c++) {
                float x = inp_bt[c] - m;
                v += x * x;
            }
            v /= C;
            float rstd_t = 1.0f / sqrtf(v + eps);
            float* out_bt = out + b * T * C + t * C;
            for (int c = 0; c < C; c++) {
                out_bt[c] = weight[c] * (inp_bt[c] - m) * rstd_t + bias[c];
            }
            mean[b * T + t] = m;
            rstd[b * T + t] = rstd_t;
        }
    }
}

void attention_forward(float* out, float* inp, int B, int T, int C, int NH) {
    // out: (B, T, C)
    // inp: (B, T, 3*C)
    int hs = C / NH;
    int C3 = 3*C;
    float scale = 1.0f / sqrtf(hs);
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float att[T];
                // 1. query_t dot key_t2
                float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        att[t2] = 0.0f;
                        float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C;
                        for (int i = 0; i < hs; i++) {
                            att[t2] += query_t[i] * key_t2[i];
                        }
                        att[t2] *= scale;
                    } else {
                        att[t2] = 0.0f;
                    }
                }
                // 2. softmax
                float exp_sum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    exp_sum += expf(att[t2]);
                }
                float exp_inv = (exp_sum == 0.0f ? 0.0f : 1.0f / exp_sum);
                for (int t2 = 0; t2 <= t; t2++) {
                    att[t2] = expf(att[t2]) * exp_inv;
                }
                // 3. out <- att[t2] * value_t2
                float* out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
                for (int t2 = 0; t2 <= t; t2++) {
                    float att_t2 = att[t2];
                    float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + 2*C;
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_t2 * value_t2[i];
                    }
                }
            }
        }
    }
}

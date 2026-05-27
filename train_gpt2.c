#include <stdio.h>
#include <math.h>
#include <float.h>

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

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

void attention_forward(float* out, float* inp, float* preatt, float* att,
                       int B, int T, int C, int NH) {
    // out: (B, T, C)
    // inp: (B, T, C*3)
    // preatt, att: (B, NH, T, T)
    int C3 = C*3;
    int hs = C / NH;
    float scale = 1.0f / sqrtf(hs);
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                // 1. query_t dot key_t2 and maxval
                float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                float* preatt_bth = preatt + b * NH*T*T + h * T*T + t * T;
                float* att_bth = att + b * NH*T*T + h * T*T + t * T;

                float maxval = -FLT_MAX;
                for (int t2 = 0; t2 <= t; t2++) {
                    float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C;
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if (val > maxval) {
                        maxval = val;
                    }
                    preatt_bth[t2] = val;
                }
                // 2. softmax
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = (expsum == 0.0f) ? 0.0f : 1.0f / expsum;
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        // PyTorchでデバッグする用のゼロ埋め
                        att_bth[t2] = 0.0f;
                    }
                }
                // 3. out_bth <- attbth_t2 * val_t2
                float* out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
                for (int t2 = 0; t2 <= t; t2++) {
                    float* val_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2;
                    float attbth_t2 = att_bth[t2];
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += attbth_t2 * val_t2[i];
                    }
                }
            }
        }
    }
}

void matmul_forward_naive(float* out, const float* inp,
                            const float* weight, const float* bias,
                            int B, int T, int C, int OC) {
    // out: (B, T, OC)
    // inp: (B, T, C)
    // weight: (OC, C)
    // bias: (OC)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int o = 0; o < OC; o++) {
                float val = (bias == NULL) ? 0.0f : bias[o];
                const float* inp_bt = inp + b * T * C + t * C;
                for (int c = 0; c < C; c++) {
                    val += inp_bt[c] * weight[o * C + c];
                }
                float* out_bt = out + b * T * OC + t * OC;
                out_bt[o] = val;
            }
        }
    }
}

void matmul_forward(float* out, const float* inp,
                    const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    // out: (B, T, OC)
    // inp: (B, T, C)
    // weight: (OC, C)
    // bias: (OC)
    const int LOOP_UNROLL = 8;
    if (B * T % LOOP_UNROLL != 0) {
        matmul_forward_naive(out, inp, weight, bias, B, T, C, OC);
        return;
    }
    for (int obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        for (int o = 0; o < OC; o++) {
            float result[LOOP_UNROLL];
            for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                result[ibt] = (bias == NULL) ? 0.0f : bias[o];
            }
            for (int c = 0; c < C; c++) {
                float w = weight[o * C + c];
                for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                    result[ibt] += inp[(obt + ibt) * C + c] * w;
                }
            }
            for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                out[(obt + ibt) * OC + o] = result[ibt];
            }
        }
    }
}

// 役割：ニューラルネットワークの層が深くなると勾配が消失する問題->近道を作ることで緩和する
void residual_forward(float* out, float* inp1, float* inp2, int N) {
    // out, inp1, inp2: (B, T, C)
    // inp1: 残差ストリーム, inp2: サブレイヤーの出力
    // N: B*T*C
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

// GeLUを要素単位で適用する非線形活性化関数
// 役割:自然言語という非線形問題は線形変換を繰り返すだけでは解くことができない
// そのため、線形変換と非線形変換を組み合わせることで、複雑な関係を近似できる
// 負の領域で勾配が0になるReLUより、負の領域でも勾配が残るGeLUを採用
void gelu_forward(float* out, float* inp, int N) {
    // out, inp: (B, T, C)
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    }
}

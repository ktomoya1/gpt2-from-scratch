#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <assert.h>

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

typedef struct {
    int max_seq_len; // 入力トークンの最大個数。
    int vocab_size; // 語彙の種類数。
    int padded_vocab_size; // vocab_sizeを128の倍数に切り上げた値、CUDAカーネル効率化のため。
    int num_layers; // Transformerのブロック数。
    int num_heads; // 並列で実行されるAttention関数の個数。
    int channels; // 各トークンのベクトルの要素数。
} GPT2Config;

#define NUM_PARAMETER_TENSORS 16
typedef struct {
    float* wte; // 埋め込み行列(Vp, C)、トークンIDをベクトルに変換する。logits計算時にwte^Tとして再利用する
    float* wpe; // 位置ベクトル(maxT, C)。トークン列を並列で処理するため、順序を持たない。そのため、位置情報を加える。
    float* ln1w; // 一層目LNのスケールパラメータ(L, C)。正規化後の各チャンネルを再スケーリングする。
    float* ln1b; // ln1wのバイアス項(L, C)。
    float* qkvw; // 入力トークンからQKVを生成するための重み行列(L, C*3, C)。
    float* qkvb; // qkvwのバイアス項(L, C*3)。
    float* attprojw; // 重み行列(L, C, C)。Multi-headAttentionの出力を線形射影して次層に渡す。
    float* attprojb; // attprojwのバイアス項(L, C)
    float* ln2w; // 二層目LNのスケールパラメータ(L, C)。正規化後の各チャンネルを再スケーリングする。
    float* ln2b; // ln2wのバイアス項(L, C)。
    float* fcw; // FFN層の重み行列(L, C*4, C)。次元を４倍に拡張し、表現力をあげる。
    float* fcb; // fcwのバイアス項(L, C*4)
    float* fcprojw; // GeLU関数後に次元を元に戻す重み行列(L, C, C*4)。
    float* fcprojb; // fcprojwのバイアス項(L, C)。
    float* lnfw; // 最終層のLN（C)。L層のresidual加算後、分散が大きくなった表現を正規化してからlogits計算に渡す。
    float* lnfb; // lnfwのバイアス項(C)。
} ParameterTensors;

#define NUM_ACTIVATION_TENSORS 23
typedef struct {
    // *_mean: 各位置が持つ平均。backwardの勾配計算時に使用
    // *_rstd: 各位置が持つ逆標準偏差。backwardで使用
    float* encoded; // (B, T, C), 埋め込みベクトル＋位置ベクトル
    float* ln1; // (L, B, T, C), トークンの特徴量を平均0, 分散1に正規化することで、どの層でも安定した分布を維持する
    float* ln1_mean; // (L, B, T)
    float* ln1_rstd; // (L, B, T)
    float* qkv; // (L, B, T, C*3), クエリ、キー、バリュー。QKの注目度に応じたVとの加重和を求める。
    float* atty; // (L, B, T, C), 注目度で重み付けしたvalueの加重和。
    float* preatt; // (L, B, NH, T, T), attention前の各位置の全位置に対する注目度。backwardで使用
    float* att; // (L, B, NH, T, T), attention後の各位置の全位置に対する注目度。backwardで使用
    float* attproj; // (L, B, T, C), 全ヘッドのattention出力を線形射影したもの。
    float* residual2; // (L, B, T, C), 残差接続。層が深くなっても勾配消失を起こさないための工夫
    float* ln2; // (L, B, T, C), FFN前の正規化。
    float* ln2_mean; // (L, B, T)
    float* ln2_rstd; // (L, B, T)
    float* fch; // (L, B, T, C*4), gelu前にベクトルを表現力向上のため一時的に拡張する。
    float* fch_gelu; // (L, B, T, C*4), 線形層に非線形活性化関数geluを入れることで非線形問題に対応。
    float* fcproj; // (L, B, T, C), 一時的に拡張したベクトルを元に戻す
    float* residual3; // (L, B, T, C), 残差接続。
    float* lnf; // (B, T, C), Transformerブロック通過後の最後の正規化。
    float* lnf_mean; // (B, T)
    float* lnf_rstd; // (B, T)
    float* logits; // (B, T, Vp), softmax関数を適用する前の語彙ごとの値。
    float* probs; // (B, T, Vp), softmax関数を適用して得た確率分布。
    float* losses; // (B, T), 次トークンを予測して得られる損失。
} ActivationTensors;

// GPT2モデルを動かすのに必要なフィールドを１箇所にまとめたもの
typedef struct {
    // GPT2モデルの定義
    GPT2Config config;

    // パラメータ（学習で更新される重み）
    ParameterTensors params; // 各パラメータ配列へのポインタ集
    size_t param_sizes[NUM_PARAMETER_TENSORS]; // 各パラメータ配列の要素数
    float* params_memory; // パラメータのデータ配列
    size_t num_parameters; // params_memoryに確保されたfloatの総数

    // forwardの中間結果
    ActivationTensors acts; // 各中間結果配列へのポインタ集
    size_t act_sizes[NUM_ACTIVATION_TENSORS]; // 各中間結果配列の要素数
    float* acts_memory; // 中間結果のデータ配列
    size_t num_activations; // acts_memoryに確保されたfloatの総数

    // backwardの勾配
    ParameterTensors grads; // 勾配配列（パラメータ）へのポインタ集
    float* grads_memory; // 勾配（パラメータ）のデータ配列
    ActivationTensors grads_acts; // 勾配配列（中間結果）へのポインタ集
    float* grads_acts_memory; // 勾配（中間結果）のデータ配列

    // optimizerの状態
    float* m_memory; // 一次モーメントのデータ
    float* v_memory; // 二次モーメントのデータ

    // 実行時の入出力管理
    int* inputs; // 現在トークンIDの配列
    int* targets; // 次トークンIDの配列
    float mean_loss; // １バッチの平均損失。backwardの起点になる
    int batch_size; // 現在のforwardで処理するバッチサイズ（acts再確保の判断に使用）
    int seq_len; // 現在のforwardで処理するトークン数（acts再確保の判断に使用）
} GPT2;

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

// 役割：各トークンが他のトークンに対してどれだけ注目するかを求め、
// 注目度でvalueを加重和することで文脈を含んだ表現に変換する
void attention_forward(float* out, float* preatt, float* att,
                       float* inp, int B, int T, int C, int NH) {
    // out: (B, T, C)
    // inp: (B, T, C*3), Q, K, Vを連結。matmul_forwardを1回で済ませるため
    // preatt, att: (B, NH, T, T), backwardで使用
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
                // 3. out <- sofmax() * val
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

// logitsにsoftmax関数を適用して確率分布を作る
void softmax_forward(float* probs, float* logits, int B, int T, int V, int Vp) {
    // probs, logits: (B, T, Vp)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* probs_bt = probs + b * T * Vp + t * Vp;
            float* logits_bt = logits + b * T * Vp + t * Vp;

            // maxvalの計算
            float maxval = -FLT_MAX;
            for (int v = 0; v < V; v++) {
                if (logits_bt[v] > maxval) maxval = logits_bt[v];
            }
            // softmaxの計算
            float expsum = 0.0f;
            for (int v = 0; v < V; v++) {
                probs_bt[v] = expf(logits_bt[v] - maxval);
                expsum += probs_bt[v];
            }
            // [V, Vp)はパディングなのでゼロ埋めする
            for (int v = 0; v < Vp; v++) {
                probs_bt[v] = (v < V) ? probs_bt[v] / expsum : 0.0f;
            }
        }
    }
}

// 各トークンのcross-entropyロスを計算する
void crossentropy_forward(float* losses, float* probs, int* targets,
                          int B, int T, int Vp) {
    // losses: (B, T), 学習する際の損失
    // probs: (B, T, Vp), 語彙の確率分布
    // targets: (B, T), 各トークンの正解トークンインデックス
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* probs_bt = probs + b * T * Vp + t * Vp;
            // 正解トークンのインデックスを取り出す
            int ix = targets[b * T + t];
            // cross-entropyロス=正解トークンの負の対数確率
            losses[b * T + t] = -logf(probs_bt[ix]);
        }
    }
}

// 各パラメータが占めるfloat個数をparam_sizesに書き込む
void fill_in_parameter_sizes(size_t* param_sizes, GPT2Config config) {
    size_t Vp = config.padded_vocab_size;
    size_t maxT = config.max_seq_len;
    size_t L = config.num_layers;
    size_t C = config.channels;

    param_sizes[0] = Vp * C; // wte
    param_sizes[1] = maxT * C; // wpe
    param_sizes[2] = L * C; // ln1w
    param_sizes[3] = L * C; // ln1b
    param_sizes[4] = L * C*3 * C; // qkvw
    param_sizes[5] = L * C*3; // qkvb
    param_sizes[6] = L * C * C; // attprojw
    param_sizes[7] = L * C; // attprojb
    param_sizes[8] = L * C; // ln2w
    param_sizes[9] = L * C; // ln2b
    param_sizes[10] = L * C*4 * C; // fcw
    param_sizes[11] = L * C*4; // fcb
    param_sizes[12] = L * C * C*4; // fcprojw
    param_sizes[13] = L * C; // fcprojb
    param_sizes[14] = C; // lnfw
    param_sizes[15] = C; // lnfb
}

// 各中間結果が占めるfloat個数をact_sizesに書き込む
void fill_in_activation_sizes(size_t* act_sizes, GPT2Config config, int B, int T) {
    size_t Vp = config.padded_vocab_size;
    size_t L = config.num_layers;
    size_t NH = config.num_heads;
    size_t C = config.channels;

    act_sizes[0] = B * T * C; // encoded
    act_sizes[1] = L * B * T * C; // ln1
    act_sizes[2] = L * B * T; // ln1_mean
    act_sizes[3] = L * B * T; // ln1_rstd
    act_sizes[4] = L * B * T * 3*C; // qkv
    act_sizes[5] = L * B * T * C; // atty
    act_sizes[6] = L * B * NH * T * T; // preatt
    act_sizes[7] = L * B * NH * T * T; // att
    act_sizes[8] = L * B * T * C; // attproj
    act_sizes[9] = L * B * T * C; // residual2
    act_sizes[10] = L * B * T * C; // ln2
    act_sizes[11] = L * B * T; // ln2_mean
    act_sizes[12] = L * B * T; // ln2_rstd
    act_sizes[13] = L * B * T * 4*C; // fch
    act_sizes[14] = L * B * T * 4*C; // fch_gelu
    act_sizes[15] = L * B * T * C; // fcproj
    act_sizes[16] = L * B * T * C; // residual3
    act_sizes[17] = B * T * C; // lnf
    act_sizes[18] = B * T; // lnf_mean
    act_sizes[19] = B * T; // lnf_rstd
    act_sizes[20] = B * T * Vp; // logits
    act_sizes[21] = B * T * Vp; // probs
    act_sizes[22] = B * T; // losses
}

// params_memoryのメモリ確保＋ParameterTensorsのフィールドポインタを割り当てる
float* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes) {
    // floatの総要素数算出
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }
    
    // 連続メモリとして一括確保することでキャッシュ効率を確保する
    float* params_memory = (float*)mallocCheck(num_parameters * sizeof(float));

    float** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw,
        &params->fcb, &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };

    float* params_memory_iterator = params_memory;
    // params_memoryをparam_sizes[i]ずつ区切って、各フィールドに対応する先頭アドレスを割り当てる
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        // *(ptrs[i]) = iterator で各フィールドのポインタ値を書き換える
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }
    return params_memory;
}

// acts_memoryのメモリ確保＋ActivationTensorsのフィールドポインタを割り当てる
float* malloc_and_point_activations(ActivationTensors* acts, size_t* act_sizes) {
    size_t num_activations = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        num_activations += act_sizes[i];
    }

    float* acts_memory = (float*)mallocCheck(num_activations * sizeof(float));
    
    float** ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->qkv,
        &acts->atty, &acts->preatt, &acts->att, &acts->attproj, &acts->residual2,
        &acts->ln2, &acts->ln2_mean, &acts->ln2_rstd, &acts->fch, &acts->fch_gelu,
        &acts->fcproj, &acts->residual3, &acts->lnf, &acts->lnf_mean, &acts->lnf_rstd,
        &acts->logits, &acts->probs, &acts->losses
    };
    float* acts_memory_iterator = acts_memory;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        *(ptrs[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }

    return acts_memory;
}

// model->actsのmalloc＋フォワードの処理＋いくつかの変数をbackward用にキャッシュ
void gpt2_forward(GPT2* model, int* inputs, int* targets, size_t B, size_t T) {
    // 推論時にはtargetsがNULLになる
    // params_memoryはgpt2_build_from_checkpointで設定される
    // NULLなら未初期化なのでエラー終了
    if (model->params_memory == NULL) {
        printf("Error1");
        exit(1);
    }

    // 便利変数展開
    size_t V = model->config.vocab_size;
    size_t Vp = model->config.padded_vocab_size;
    size_t L = model->config.num_layers;
    size_t NH = model->config.num_heads;
    size_t C = model->config.channels;

    // トークンIDが[0, V)範囲内かチェック
    for (size_t i = 0; i < B * T; i++) {
        assert(0 <= inputs[i] && inputs[i] < V);
        if (targets != NULL) {
            assert(0 <= targets[i] && targets[i] < V);
        }
    }

    // actsメモリを初回のみ確保
    if (model->acts_memory == NULL) {
        model->batch_size = B;
        model->seq_len = T;

        fill_in_activation_sizes(model->act_sizes, model->config, (int)B, (int)T);
        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) { num_activations += model->act_sizes[i]; }
        model->num_activations = num_activations;
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);

        model->inputs = (int*)mallocCheck(B * T * sizeof(int));
        model->targets = (int*)mallocCheck(B * T * sizeof(int));
    } else {
        if (B != model->batch_size || T != model->seq_len) {
            printf("Error2");
            exit(1);
        }
    }

    // backward用のキャッシュ
    memcpy(model->inputs, inputs, B * T * sizeof(int));
    if (targets != NULL) {
        memcpy(model->targets, targets, B * T * sizeof(int));
    }

    // フォワード処理
    ParameterTensors params = model->params;
    ActivationTensors acts = model->acts;
    float *residual;

    encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C);
    for (int l = 0; l < L; l++) {
        residual = (l == 0) ? acts.encoded : acts.residual3 + (l - 1) * B * T * C;

        // l番目のレイヤの重み・アクティベーションポインタをオフセットで取り出す
        float* l_ln1w = params.ln1w + l * C;
        float* l_ln1b = params.ln1b + l * C;
        float* l_qkvw = params.qkvw + l * 3*C * C;
        float* l_qkvb = params.qkvb + l * 3*C;
        float* l_attprojw = params.attprojw + l * C * C;
        float* l_attprojb = params.attprojb + l * C;
        float* l_ln2w = params.ln2w + l * C;
        float* l_ln2b = params.ln2b + l * C;
        float* l_fcw = params.fcw + l * 4*C * C;
        float* l_fcb = params.fcb + l * 4*C;
        float* l_fcprojw = params.fcprojw + l * C * 4*C;
        float* l_fcprojb = params.fcprojb + l * C;

        float* l_ln1 = acts.ln1 + l * B * T * C;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float* l_qkv = acts.qkv + l * B * T * 3*C;
        float* l_atty = acts.atty + l * B * T * C;
        float* l_preatt = acts.preatt + l * B * NH * T * T;
        float* l_att = acts.att + l * B * NH * T * T;
        float* l_attproj = acts.attproj + l * B * T * C;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float* l_fch = acts.fch + l * B * T * 4*C;
        float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        float* l_fcproj = acts.fcproj + l * B * T * C;
        float* l_residual3 = acts.residual3 + l * B * T * C;

        layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
        matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
        attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        residual_forward(l_residual2, residual, l_attproj, B * T * C);
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
        gelu_forward(l_fch_gelu, l_fch, B * T * 4*C);
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
        residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C);
    }

    // 最終レイヤの出力に対して最終LN->logits->softmaxを適用
    residual = acts.residual3 + (L - 1) * B * T * C;
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
    matmul_forward(acts.logits, acts.lnf, params.wte, NULL, B, T, C, Vp);
    softmax_forward(acts.probs, acts.logits, B, T, V, Vp);

    // mean_lossの計算。学習進捗の確認用に使用
    if (targets != NULL) {
        crossentropy_forward(acts.losses, acts.probs, targets, B, T, Vp);
        float mean_loss = 0.0f;
        for (size_t i = 0; i < B * T; i++) {
            mean_loss += acts.losses[i];
        }
        mean_loss /= B * T;
        model->mean_loss = mean_loss;
    } else {
        // 推論時はmean_lossは使用しない
        model->mean_loss = -1.0f;
    }
}

// grads_memoryとgrads_acts_memoryのゼロクリア。
// backwardで勾配を累積するため、前ステップの値が残ると誤った勾配になる。
// 両メモリはgpt2_backward内で遅延確保されるため、NULLチェックしてからクリアする。
void gpt2_zero_grad(GPT2* model) {
    if (model->grads_memory != NULL) {
        memset(model->grads_memory, 0, model->num_parameters * sizeof(float));
    }
    if (model->grads_acts_memory != NULL) {
        memset(model->grads_acts_memory, 0, model->num_activations * sizeof(float));
    }
}

// softmax+cross-entropyの合成関数に対し、logitsの勾配∂E/∂ziを計算する
// ∂E/∂zi = (pi - 1[i == ix])/(B・T)
void crossentropy_softmax_backward(float* dlogits, float* dlosses, float* probs, int* targets,
                                   int B, int T, int V, int Vp) {
    // dlogits, probs: (B, T, Vp)
    // dlosses, targets: (B, T)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* probs_bt = probs + b * T * Vp + t * Vp;
            float* dlogits_bt = dlogits + b * T * Vp + t * Vp;
            // 上流勾配1/(B・T)
            float dloss = dlosses[b * T + t];
            int ix = targets[b * T + t];
            for (int v = 0; v < V; v++) {
                float indicator = (v == ix) ? 1.0f : 0.0f;
                dlogits_bt[v] += dloss * (probs_bt[v] - indicator);
            }
        }
    }
}

// 上流から流れてきた勾配をinp1, inp2に加算する
// ∂E/∂inp = ∂E/∂out・∂out/∂inp = dout・1 = dout
void residual_backward(float* dinp1, float* dinp2, float* dout, int N) {
    for (int i = 0; i < N; i++) {
        // 複数経路から勾配が流れ込む可能性があるため、加算で蓄積する
        dinp1[i] += dout[i];
        dinp2[i] += dout[i];
    }
}

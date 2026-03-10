import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# Configuracoes globais
# ─────────────────────────────────────────────
np.random.seed(42)

D_MODEL  = 64    # paper usa 512; reduzido para CPU conforme instrucao
D_FF     = 256   # paper usa 2048; proporcao 4x mantida
N_LAYERS = 6

# ─────────────────────────────────────────────
# Passo 1 - Preparacao dos dados
# ─────────────────────────────────────────────

vocab_dict = {
    "o": 0, "banco": 1, "bloqueou": 2, "cartao": 3,
    "de": 4, "credito": 5, "meu": 6, "por": 7,
    "suspeita": 8, "fraude": 9
}
vocab_df = pd.DataFrame(list(vocab_dict.items()), columns=["palavra", "id"])

frase     = ["o", "banco", "bloqueou", "meu", "cartao"]
token_ids = [vocab_dict[p] for p in frase]

vocab_size      = len(vocab_dict)
embedding_table = np.random.randn(vocab_size, D_MODEL)

X_embed = embedding_table[token_ids]    # (seq_len, d_model)
X       = X_embed[np.newaxis, :, :]    # (1, seq_len, d_model)

print("=== Preparacao dos Dados ===")
print(vocab_df.to_string(index=False))
print(f"\nFrase     : {frase}")
print(f"Token IDs : {token_ids}")
print(f"Tensor X  : {X.shape}  (batch, seq_len, d_model)\n")


# ─────────────────────────────────────────────
# Passo 2 - Funcoes matematicas auxiliares
# ─────────────────────────────────────────────

def softmax(x):
    """Softmax numericamente estavel ao longo do ultimo eixo."""
    x_max = np.max(x, axis=-1, keepdims=True)
    e_x   = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def layer_norm(x, epsilon=1e-6):
    """Normalizacao por camada operando no ultimo eixo (features)."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var  = np.var(x,  axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + epsilon)


def relu(x):
    return np.maximum(0, x)


# ─────────────────────────────────────────────
# Passo 2.1 - Scaled Dot-Product Attention
# ─────────────────────────────────────────────

class SelfAttention:
    """
    Attention(Q, K, V) = softmax( Q K^T / sqrt(dk) ) V

    Cada instancia carrega seus proprios pesos WQ, WK, WV,
    um conjunto por camada do Encoder.
    """

    def __init__(self, d_model):
        self.WQ = np.random.randn(d_model, d_model) * 0.01
        self.WK = np.random.randn(d_model, d_model) * 0.01
        self.WV = np.random.randn(d_model, d_model) * 0.01

    def forward(self, X):
        Q  = X @ self.WQ                               # (batch, seq, d_model)
        K  = X @ self.WK
        V  = X @ self.WV
        dk = Q.shape[-1]

        scores            = Q @ K.transpose(0, 2, 1) / np.sqrt(dk)
        attention_weights = softmax(scores)            # (batch, seq, seq)
        return attention_weights @ V                   # (batch, seq, d_model)


# ─────────────────────────────────────────────
# Passo 2.3 - Feed-Forward Network
# ─────────────────────────────────────────────

class FeedForward:
    """
    FFN(x) = max(0, x W1 + b1) W2 + b2

    Expande d_model -> d_ff com ReLU, depois contrai de volta.
    """

    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        hidden = relu(x @ self.W1 + self.b1)    # (batch, seq, d_ff)
        return hidden @ self.W2 + self.b2        # (batch, seq, d_model)


# ─────────────────────────────────────────────
# Bloco completo - EncoderLayer
# ─────────────────────────────────────────────

class EncoderLayer:
    """
    Uma camada completa do Encoder:
        1. X_att   = SelfAttention(X)
        2. X_norm1 = LayerNorm(X + X_att)
        3. X_ffn   = FFN(X_norm1)
        4. X_out   = LayerNorm(X_norm1 + X_ffn)
    """

    def __init__(self, d_model, d_ff):
        self.attention = SelfAttention(d_model)
        self.ffn       = FeedForward(d_model, d_ff)

    def forward(self, X):
        X_att   = self.attention.forward(X)
        X_norm1 = layer_norm(X + X_att)

        X_ffn   = self.ffn.forward(X_norm1)
        X_out   = layer_norm(X_norm1 + X_ffn)
        return X_out


# ─────────────────────────────────────────────
# Passo 3 - Encoder: empilhando N=6 camadas
# ─────────────────────────────────────────────

class TransformerEncoder:
    """
    Pilha de N camadas identicas do Encoder.
    O output de cada camada e o input da proxima.
    """

    def __init__(self, n_layers, d_model, d_ff):
        self.layers = [EncoderLayer(d_model, d_ff) for _ in range(n_layers)]

    def forward(self, X):
        print("=== Forward Pass pelo Encoder ===")
        print(f"Entrada (Camada 1): {X.shape}")

        for i, layer in enumerate(self.layers, start=1):
            X = layer.forward(X)
            print(f"Saida da Camada {i}: {X.shape}")

        return X


# ─────────────────────────────────────────────
# Execucao principal
# ─────────────────────────────────────────────

encoder = TransformerEncoder(N_LAYERS, D_MODEL, D_FF)
Z = encoder.forward(X)

print("\n=== Validacao de Sanidade ===")
print(f"Shape de entrada : (1, {len(frase)}, {D_MODEL})")
print(f"Shape de saida Z : {Z.shape}")
assert Z.shape == X_embed[np.newaxis].shape, "ERRO: dimensoes incompativeis!"
print("Dimensoes preservadas corretamente apos 6 camadas.")
print(f"\nPrimeiro vetor Z (token '{frase[0]}'):")
print(Z[0, 0, :8], "...")

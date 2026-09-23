"""Rede neural com camada escondida configurável: N neurônios e ativação trocável.

Versão generalizada da rede 2-2-1 do main.py, que tem sigmoid fixa e o
backpropagation escrito nó a nó. Aqui o forward e o backward são reescritos em
forma matricial: a mesma regra da cadeia, promovida de escalares para matrizes.
Isso permite variar o número de neurônios da camada escondida e a função de
ativação sem reescrever nenhuma derivada.

Correspondência com o main.py (ver plano-alteracao-rede-neural.md):

    main.py (2-2-1, explícito)                  aqui (matricial)
    v0 = w0[0,0]*x[0] + w0[0,1]*x[1] + b0[0]    V0 = X @ W0.T + b0
    y0 = sigmoid(v0)                            Y0 = f(V0)
    v2 = y0*w1[0] + y1*w1[1] + b1[0]            V1 = Y0 @ W1 + b1[0]
    grad_v2 = grad_e * y2 * (1 - y2)            dV1 = (E/N) * Y1 * (1 - Y1)
    grad_w1[i] = grad_s2i * yi                  gW1 = Y0.T @ dV1
    grad_y0 = grad_v2 * w1[0]                   dY0 = np.outer(dV1, W1)
    grad_v0 = grad_y0 * y0 * (1 - y0)           dV0 = dY0 * df(V0, Y0)
    grad_w0[i,j] = grad_sij * x[j]              gW0 = dV0.T @ X

O laço sobre as 100 amostras (main.py linhas 130-137) some: a soma passa a ser
feita pelos próprios produtos de matrizes.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

SEED = 42
EPOCAS = 10000

# H, ativação da camada escondida e taxa de aprendizado.
# A primeira linha reproduz o main.py: lá os gradientes das 100 amostras são
# somados e a taxa é 0.1, o que equivale a usar o gradiente médio com taxa 10.
CONFIGS = [
    {'H': 2,  'ativacao': 'sigmoid', 'taxa': 10.0},
    {'H': 4,  'ativacao': 'sigmoid', 'taxa': 10.0},
    {'H': 8,  'ativacao': 'sigmoid', 'taxa': 10.0},
    {'H': 8,  'ativacao': 'tanh',    'taxa': 1.0},
    {'H': 8,  'ativacao': 'relu',    'taxa': 0.1},
    {'H': 16, 'ativacao': 'tanh',    'taxa': 1.0},
]


# --------------------------------------------------------------------------
# ativações e suas derivadas
# --------------------------------------------------------------------------

def sigmoid(v):
    """Sigmoid em forma estável: não estoura o exp para v muito negativo."""
    return np.exp(np.minimum(v, 0)) / (1 + np.exp(-np.abs(v)))


def d_sigmoid(v, y):
    return y * (1 - y)


def tanh(v):
    return np.tanh(v)


def d_tanh(v, y):
    return 1 - y ** 2


def relu(v):
    return np.maximum(0.0, v)


def d_relu(v, y):
    return (v > 0).astype(float)


# Cada derivada recebe a pré-ativação V e a pós-ativação Y, e usa a que for mais
# conveniente: sigmoid e tanh se expressam por Y, ReLU precisa do sinal de V.
ATIVACOES = {
    'sigmoid': (sigmoid, d_sigmoid),
    'tanh': (tanh, d_tanh),
    'relu': (relu, d_relu),
}


# --------------------------------------------------------------------------
# rede
# --------------------------------------------------------------------------

def init_params(n_entradas, H, ativacao, rng):
    """Pesos centrados em zero (Xavier/He), vieses zerados.

    O main.py usa np.random.rand, que devolve valores só em [0, 1). Com H=2 isso
    passa, mas com H maior todas as unidades escondidas começam parecidas e
    empurrando para o mesmo lado, desperdiçando a capacidade extra.
    """
    escala = np.sqrt(2.0 / n_entradas) if ativacao == 'relu' else np.sqrt(1.0 / n_entradas)
    return {
        'W0': rng.standard_normal((H, n_entradas)) * escala,
        'b0': np.zeros(H),
        'W1': rng.standard_normal(H) * np.sqrt(1.0 / H),
        'b1': np.zeros(1),
    }


def forward(X, params, f):
    """X tem shape (N, 2). Devolve as pré-ativações e ativações das duas camadas."""
    V0 = X @ params['W0'].T + params['b0']   # (N, H)
    Y0 = f(V0)                               # (N, H)  <- ativação configurável
    V1 = Y0 @ params['W1'] + params['b1'][0]  # (N,)
    Y1 = sigmoid(V1)                         # (N,)  saída é sempre sigmoid: rótulos 0/1
    return V0, Y0, V1, Y1


def backward(X, Y, params, f, df):
    """Gradientes da perda média e a própria perda."""
    N = X.shape[0]
    V0, Y0, _, Y1 = forward(X, params, f)
    E = Y1 - Y
    L = 0.5 * np.mean(E ** 2)

    dV1 = (E / N) * Y1 * (1 - Y1)            # (N,)
    dY0 = np.outer(dV1, params['W1'])        # (N, H)
    dV0 = dY0 * df(V0, Y0)                   # (N, H)  <- só esta linha depende da ativação

    grads = {
        'b1': np.array([dV1.sum()]),
        'W1': Y0.T @ dV1,                    # (H,)
        'b0': dV0.sum(axis=0),               # (H,)
        'W0': dV0.T @ X,                     # (H, 2)
    }
    return grads, L


def treinar(X, Y, params, ativacao, taxa, epocas=EPOCAS):
    """Gradiente descendente em batch completo. Não altera os params recebidos."""
    f, df = ATIVACOES[ativacao]
    params = {k: v.copy() for k, v in params.items()}
    historico = []
    for _ in range(epocas):
        grads, L = backward(X, Y, params, f, df)
        for k in params:
            params[k] -= taxa * grads[k]
        historico.append(L)
    return params, historico


def prever(X, params, f):
    _, _, _, Y1 = forward(X, params, f)
    return (Y1 > 0.5).astype(int)


def acuracia(X, Y, params, f):
    return float(np.mean(prever(X, params, f) == Y))


# --------------------------------------------------------------------------
# verificação do backward
# --------------------------------------------------------------------------

def verificar_gradiente(ativacao='sigmoid', H=3, N=5, eps=1e-5):
    """Compara o gradiente analítico com diferenças finitas centrais.

    É o que de fato prova que o backward está correto, independente de a rede
    aprender bem ou não. Devolve o erro relativo máximo sobre todos os
    parâmetros.
    """
    rng = np.random.default_rng(0)
    f, df = ATIVACOES[ativacao]
    X = rng.standard_normal((N, 2))
    Y = rng.integers(0, 2, N).astype(float)
    params = init_params(2, H, ativacao, rng)

    grads, _ = backward(X, Y, params, f, df)

    erro_max = 0.0
    for nome, P in params.items():
        for idx in np.ndindex(P.shape):
            original = P[idx]

            P[idx] = original + eps
            _, L_mais = backward(X, Y, params, f, df)
            P[idx] = original - eps
            _, L_menos = backward(X, Y, params, f, df)
            P[idx] = original

            numerico = (L_mais - L_menos) / (2 * eps)
            analitico = grads[nome][idx]
            denominador = max(abs(numerico), abs(analitico), 1e-12)
            erro_max = max(erro_max, abs(numerico - analitico) / denominador)
    return erro_max


# --------------------------------------------------------------------------
# saídas
# --------------------------------------------------------------------------

def tabela_comparativa(resultados):
    cabecalho = (f"{'H':>3}  {'ativação':<9}  {'taxa':>5}  "
                 f"{'perda final':>12}  {'acc antes':>10}  {'acc depois':>11}")
    print(cabecalho)
    print('-' * len(cabecalho))
    for r in resultados:
        print(f"{r['H']:>3}  {r['ativacao']:<9}  {r['taxa']:>5.1f}  "
              f"{r['historico'][-1]:>12.6f}  {r['acc_antes']:>10.2f}  {r['acc_depois']:>11.2f}")


def plotar_fronteiras(resultados, X, Y, arquivo='fronteiras.svg'):
    """Superfície de decisão de cada configuração, com os pontos por cima."""
    cores = ['blue' if k == 0 else 'red' for k in Y]

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    # a grade entra no forward como um lote comum de shape (N, 2)
    grade = np.c_[xx.ravel(), yy.ravel()]

    colunas = 3
    linhas = int(np.ceil(len(resultados) / colunas))
    fig, axes = plt.subplots(linhas, colunas,
                             figsize=(5 * colunas, 4 * linhas), squeeze=False)
    eixos = axes.ravel()

    for ax, r in zip(eixos, resultados):
        f, _ = ATIVACOES[r['ativacao']]
        _, _, _, Z = forward(grade, r['params'], f)
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, levels=[0, 0.5, 1],
                    colors=['#aaccff', '#ffbbbb'], alpha=0.5)
        ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=1)
        ax.scatter(X[:, 0], X[:, 1], c=cores, s=18,
                   edgecolors='black', linewidths=0.3)
        ax.set_title(f"H={r['H']}  {r['ativacao']}  acc={r['acc_depois']:.2f}")

    for ax in eixos[len(resultados):]:
        ax.axis('off')

    fig.tight_layout()
    fig.savefig(arquivo)
    plt.close(fig)


def plotar_perdas(resultados, arquivo='perdas.svg'):
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in resultados:
        ax.plot(r['historico'], label=f"H={r['H']} {r['ativacao']}")
    ax.set_yscale('log')
    ax.set_xlabel('época')
    ax.set_ylabel('perda média (escala log)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(arquivo)
    plt.close(fig)


def modelo_keras(X, Y, H, ativacao):
    """Mesma arquitetura no Keras, para comparação.

    A taxa é a do main.py (0.1): a taxa da versão manual não é transferível,
    porque o Keras usa mini-batch e perda média, e aqui o treino é em batch
    completo. A comparação é qualitativa — o esperado é chegarem ao mesmo
    patamar de acurácia, não os números baterem.
    """
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Input
    from keras.optimizers import SGD

    keras.utils.set_random_seed(SEED)

    model = Sequential()
    model.add(Input(shape=(2,)))
    model.add(Dense(H, activation=ativacao))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error',
                  optimizer=SGD(learning_rate=0.1),
                  metrics=['accuracy'])
    model.fit(X, Y, epochs=100, verbose=False, batch_size=5)

    # o main.py atribui este retorno a `acc` e nunca imprime (linha 165)
    perda, acc = model.evaluate(X, Y, verbose=False)
    return perda, acc


# --------------------------------------------------------------------------

def main():
    print('=== verificação numérica do gradiente (diferenças finitas) ===')
    for ativacao in ATIVACOES:
        erro = verificar_gradiente(ativacao)
        print(f'  {ativacao:<8} erro relativo máximo: {erro:.3e}')
    print()

    # random_state fixo: sem ele cada execução sortearia dados diferentes e a
    # tabela mediria o sorteio, não o efeito de H
    X, Y = datasets.make_moons(100, noise=0.1, random_state=SEED)
    Y = Y.astype(float)

    resultados = []
    for cfg in CONFIGS:
        f, _ = ATIVACOES[cfg['ativacao']]
        # mesma semente para todas as configurações
        rng = np.random.default_rng(SEED)
        iniciais = init_params(X.shape[1], cfg['H'], cfg['ativacao'], rng)

        acc_antes = acuracia(X, Y, iniciais, f)
        params, historico = treinar(X, Y, iniciais, cfg['ativacao'], cfg['taxa'])
        acc_depois = acuracia(X, Y, params, f)

        resultados.append({**cfg, 'params': params, 'historico': historico,
                           'acc_antes': acc_antes, 'acc_depois': acc_depois})

    print('=== comparação entre configurações ===')
    tabela_comparativa(resultados)
    print()

    plotar_fronteiras(resultados, X, Y)
    plotar_perdas(resultados)
    print('gerados: fronteiras.svg, perdas.svg')
    print()

    melhor = max(resultados, key=lambda r: r['acc_depois'])
    print(f"=== Keras com a melhor configuração manual "
          f"(H={melhor['H']}, {melhor['ativacao']}) ===")
    perda, acc = modelo_keras(X, Y, melhor['H'], melhor['ativacao'])
    print(f'  perda: {perda:.6f}   acurácia: {acc:.2f}')
    print(f"  manual: perda {melhor['historico'][-1]:.6f}   "
          f"acurácia {melhor['acc_depois']:.2f}")


if __name__ == '__main__':
    main()

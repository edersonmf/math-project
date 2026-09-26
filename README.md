# Projeto - Matemática para Ciência de Dados

Trabalho individual: Ederson Marcos Ferreira (emf4@cin.ufpe.br)

Rede neural implementada do zero, com forward e backpropagation escritos à mão,
treinada para separar as duas luas do `make_moons` (problema não linearmente
separável).

- **`main.py`** — versão original: rede **2‑2‑1**, sigmoid fixa, backpropagation
  escrito nó a nó.
- **`rede_configuravel.py`** — versão generalizada: **N neurônios** na camada
  escondida e **função de ativação configurável**.

```bash
uv run main.py               # versão original
uv run rede_configuravel.py  # versão generalizada + comparação
```

---

## A alteração na rede

No `main.py` cada soma parcial do grafo computacional é uma variável (`s00`,
`s01`, `s02`, …), o que torna as derivadas explícitas — mas **congela a
arquitetura no código**: mudar o número de neurônios ou a ativação exigiria
reescrever as ~35 linhas do backward. E duas unidades sigmoid não dão conta das
luas: a fronteira resultante é quase uma reta.

O `rede_configuravel.py` reescreve forward e backward em **forma matricial** — a
mesma regra da cadeia, promovida de escalares para matrizes:

| `main.py` (2‑2‑1, explícito) | `rede_configuravel.py` (matricial) |
|---|---|
| `v0 = w0[0,0]*x[0] + w0[0,1]*x[1] + b0[0]` | `V0 = X @ W0.T + b0` |
| `y0 = sigmoid(v0)` | `Y0 = f(V0)` |
| `grad_v0 = grad_y0 * y0 * (1 - y0)` | `dV0 = dY0 * df(V0, Y0)` |
| `grad_w0[i,j] = grad_sij * x[j]` | `gW0 = dV0.T @ X` |

Com isso **uma única linha** do backward depende da ativação (`dV0`), então
trocar sigmoid por tanh ou ReLU não exige mexer em derivada nenhuma. O laço sobre
as 100 amostras também desaparece: a soma passa a ser feita pelos próprios
produtos de matrizes.

Três ajustes acompanham a generalização, sem os quais ela não funciona:

1. **Semente fixa.** Sem `random_state`, cada execução sorteia dados novos e a
   comparação mediria o sorteio, não o efeito de `H`.
2. **Inicialização centrada em zero** (Xavier/He) no lugar de `np.random.rand`,
   que só devolve valores positivos — com `H` maior, todas as unidades escondidas
   começariam parecidas e a capacidade extra se perderia.
3. **Gradiente médio em vez de somado.** O `main.py` soma 100 gradientes e aplica
   taxa 0.1, ou seja, passo efetivo ~100× o nominal; só não diverge porque a
   sigmoid é limitada. Equivalência: *somado com 0.1 ≡ médio com 10* — por isso a
   configuração base usa `taxa=10.0`, para reproduzir o original.

**Verificação:** o backward é conferido por diferenças finitas centrais, com erro
relativo máximo de `3.2e-09` (sigmoid), `2.1e-08` (tanh) e `3.0e-09` (relu).

---

## Comparação entre configurações

Mesma base (`make_moons`, 100 pontos, ruído 0.1) e mesma semente em todas as
linhas. A primeira reproduz o `main.py`, que dá 88/100.

| H | ativação | taxa | perda final | acc antes | acc depois |
|---|---|---|---|---|---|
| 2 | sigmoid | 10.0 | 0.040600 | 0.50 | **0.89** |
| 4 | sigmoid | 10.0 | 0.000061 | 0.55 | **1.00** |
| 8 | sigmoid | 10.0 | 0.000077 | 0.41 | **1.00** |
| 8 | tanh | 1.0 | 0.000232 | 0.34 | **1.00** |
| 8 | relu | 0.1 | 0.012802 | 0.34 | **0.97** |
| 16 | tanh | 1.0 | 0.000280 | 0.89 | **1.00** |

Em `fronteiras.svg`: com `H=2` a fronteira é quase uma reta, que atravessa o
entrelaçamento das luas; de `H=4` em diante aparece a curva em S que as
acompanha. Com ReLU a fronteira é poligonal, com cantos visíveis — composição de
funções lineares por partes.

> A acurácia é medida **no próprio conjunto de treino**, como no `main.py`. Com
> `H=16` há espaço para decorar os dados, então os `1.00` são números de ajuste,
> não de generalização.

---

## Executando com outras funções de ativação

Ativações disponíveis: **`sigmoid`**, **`tanh`**, **`relu`**.

### Alterando a lista de configurações

Edite `CONFIGS` no topo de `rede_configuravel.py` e rode o script. Ele executa
todas as linhas da lista, imprime a tabela e gera os gráficos:

```python
CONFIGS = [
    {'H': 8,  'ativacao': 'tanh', 'taxa': 1.0},
    {'H': 32, 'ativacao': 'relu', 'taxa': 0.5},
]
```

```bash
uv run rede_configuravel.py
```

### Treinando uma única configuração

Para um teste rápido, sem gerar gráficos:

```python
import numpy as np
from sklearn import datasets
from rede_configuravel import ATIVACOES, SEED, init_params, treinar, acuracia

X, Y = datasets.make_moons(100, noise=0.1, random_state=SEED)
Y = Y.astype(float)

H, ativacao, taxa = 8, 'tanh', 1.0

f, _ = ATIVACOES[ativacao]
iniciais = init_params(2, H, ativacao, np.random.default_rng(SEED))
params, historico = treinar(X, Y, iniciais, ativacao, taxa)

print(f'H={H} {ativacao}  perda={historico[-1]:.6f}  '
      f'acc={acuracia(X, Y, params, f):.2f}')
```

```
H=8 tanh  perda=0.000232  acc=1.00
```

### Sobre a taxa de aprendizado

A taxa **não é transferível entre ativações** — é o parâmetro a ajustar primeiro
se o resultado vier ruim. Valores que funcionam bem:

| ativação | taxa | motivo |
|---|---|---|
| `sigmoid` | 10.0 | gradiente limitado por `y(1-y) ≤ 0.25`, tolera passo grande |
| `tanh` | 1.0 | gradiente até 4× o da sigmoid |
| `relu` | 0.1 | gradiente não saturado; taxa alta diverge |

### Adicionando uma nova ativação

Basta registrar a função e sua derivada em `ATIVACOES`. A derivada recebe a
pré‑ativação `V` e a pós‑ativação `Y`, e usa a que for conveniente:

```python
def leaky_relu(v):
    return np.where(v > 0, v, 0.01 * v)

def d_leaky_relu(v, y):
    return np.where(v > 0, 1.0, 0.01)

ATIVACOES['leaky_relu'] = (leaky_relu, d_leaky_relu)
```

Confira a derivada antes de treinar — `verificar_gradiente('leaky_relu')` deve
devolver erro da ordem de `1e-8` ou menor.

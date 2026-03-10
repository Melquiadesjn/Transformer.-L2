# Transformer Encoder From Scratch

Implementacao do Forward Pass do Encoder do Transformer, baseado no artigo
"Attention Is All You Need" (Vaswani et al., 2017).

Disciplina: Topicos em Inteligencia Artificial - 2026.1
Professor: Prof. Dimmy Magalhaes
Instituicao: iCEV - Instituto de Ensino Superior

---

## Como Rodar

### Pre-requisitos

```
pip install numpy pandas
```

### Execucao

```
python transformer_encoder.py
```

A saida exibe as dimensoes do tensor em cada etapa e o vetor Z final apos passar
pelas 6 camadas do Encoder.

---

## Estrutura do projeto

```
transformer_encoder.py   - implementacao principal
README.md                - este arquivo
```

---

## Nota de uso de ferramentas

Durante o desenvolvimento, Claude (Anthropic) foi consultado para esclarecer
duvidas de sintaxe do numpy (broadcasting, np.matmul, np.mean/var com axis=-1).
A logica, estrutura das classes e implementacao das equacoes foram desenvolvidas
pelo autor.

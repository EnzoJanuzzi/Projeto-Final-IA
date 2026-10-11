# 🧪 Aprendizado de Máquina Explicável na Predição Energética de Nanopartículas para Catálise

Repositório dedicado ao desenvolvimento e à interpretação de modelos de *Machine Learning* aplicados a cálculos de primeiros princípios (DFT/VASP) em sistemas catalíticos de nanopartículas metálicas (Ru). O objetivo principal é mapear e explicar as relações físico-químicas não lineares que governam a estabilidade energética de intermediários reacionais (rotas *N-down* e *O-down*) utilizando técnicas de **Inteligência Artificial Explicável (XAI)**.

---

## 📌 Sumário
* [🎯 Objetivos do Projeto](#objetivos-do-projeto)
* [🛠️ Como Executar](#como-executar)
* [📊 Dicionário de Dados](#dicionario-de-dados)
* [⚙️ Pipeline do Trabalho](#pipeline-e-pre-processamento)
* [🧠 Modelos Implementados](#modelos-implementados)
* [🔍 Ferramentas de Explicabilidade (XAI)](#ferramentas-de-explicabilidade-xai)
* [📈 Principais Conclusões](#principais-conclusoes)

---
<a id="objetivos-do-projeto"></a>
## 🎯 Objetivos do Projeto

1. **Predição Energética:** Avaliar modelos supervisionados para prever a energia total (`total_energy_eV`) de intermediários catalíticos.
2. **Comparação de Paradigmas:** Contrastar modelos intrinsecamente interpretáveis (Regressão OLS com regularização e Árvores de Decisão rasas) com abordagens flexíveis aditivas (GAMs).
3. **Explicabilidade Pós-Hoc:** Analisar o impacto marginal e individual das variáveis físico-químicas sobre as decisões dos modelos através de Gráficos de Dependência Parcial (PDP) e Expectativa Condicional Individual (ICE).

---
<a id="como-executar"></a>
## 🛠️ Como Executar

### 1. Clonar o repositório e preparar o ambiente

```bash
git clone https://github.com/EnzoJanuzzi/projeto-ia-nanoparticulas.git
cd projeto-ia-nanoparticulas
python -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate

```

### 2. Instalar dependências caso não tenha

```bash
pip install numpy pandas matplotlib seaborn scikit-learn pygam statsmodels ase pymatgen

```

### 3. Executar o Notebook

Abra o Jupyter e execute as células sequencialmente:

```bash
jupyter lab "Projeto final IA.ipynb"

```

---
<a id="dicionario-de-dados"></a>
## 📊 Dicionário de Dados

Os dados foram gerados via simulações DFT (VASP) para diferentes geometrias e caminhos reacionais (*N-down* e *O-down*). Seguem alguns parâmetros extraídos:

| Coluna | Tipo | Unidade / Formato | Descrição Físico-Química |
| --- | --- | --- | --- |
| `e_fermi_eV` | Contínua | $\text{eV}$ | Energia do nível de Fermi do sistema catalítico obtida do cálculo eletrônico. |
| `d_band_center_Ru_eV` | Contínua | $\text{eV}$ | Centro da banda $d$ dos átomos de rutênio, métrica clássica para reatividade em catálise heterogênea. |
| `binding_distance_A` | Contínua | $\text{Å}$ | Distância de ligação mínima entre o adsorbato e a superfície da nanopartícula. |
| `adsorbate_total_mag` / `total_mag_muB` | Contínua | $\mu_B$ | Momento magnético total do sistema/adsorbato integrado a partir do `OUTCAR`/`OSZICAR`. |
| `adsorption_site` | Categórica | Texto (`top`, `bridge`, `hollow`) | Topologia local do sítio onde a molécula se ancorou na faceta da nanopartícula. |
| `pathway` | Categórica | Texto (`N-down`, `O-down`) | Orientação molecular de aproximação do intermediário químico à superfície. |
| `intermediate` | Categórica | Fórmula química | Espécie intermediária ligada à partícula (ex.: $\text{NO}$, $\text{HNO}$, $\text{NH}_2$, $\text{OH}$). |
| `total_energy_eV` *(Target)* | Contínua | $\text{eV}$ | Energia eletrônica total fundamental convergida pelo VASP. |

> **Nota de Pré-processamento:** Coordenadas absolutas do centro de massa (`com_x`, `com_y`, `com_z`) e índices arbitrários de átomos foram descartados para garantir a invariância translacional e rotacional do aprendizado supervisionado.

---
<a id="#pipeline-e-pre-processamento"></a>
## ⚙️ Pipeline e Pré-processamento

O fluxo utiliza Scikit-Learn estruturado via `ColumnTransformer` para evitar vazamento estatístico (*data leakage*):

```
[Dados Brutos DFT (N-down + O-down)]
       │
       ├─► Split dos dados (Train / Test)
       │
       ├─► Colunas Contínuas ────► StandardScaler (μ=0, σ=1)
       │
       └─► Colunas Categóricas ──► OneHotEncoder (drop='first' / dummies)

```

---
<a id="modelos-implementados"></a>
## 🧠 Modelos Implementados

* **Regressão Linear (OLS / ElasticNet / Ridge):** Modelo linear clássico via `statsmodels` utilizado como *baseline* interpretável. Evidencia a limitação de hipóteses puramente lineares para a complexidade da catálise:

$$y = \beta_0 + \beta_1 x_1 + \cdots + \beta_n x_n + \epsilon$$
  
* **Árvores de Decisão (`DecisionTreeRegressor`):** Modelo não paramétrico com profundidade controlada (`max_leaf_nodes=6`) para permitir inspeção direta das regras de partição dos nós, bem como modelo complexo com mais nós folha para comparação.
* **Modelos Aditivos Generalizados (GAMs - `pygam`):** Ajuste de splines penalizadas lineares $s(x)$ para as variáveis contínuas e termos lineares $l(x)$ para as variáveis binárias:

$$g(E[y|X]) = \beta_0 + f_1(x_{1}) + \cdots + f_n(x_{n}) = \beta_0 + \sum_{i=1}^n f_i(x_{i})$$

---
<a id="ferramentas-de-explicabilidade-xai"></a>
## 🔍 Ferramentas de Explicabilidade (XAI)

* **Partial Dependence Plots (PDP):** Medem o efeito médio marginal de cada variável isolada no valor previsto de energia.
* **Individual Conditional Expectation (ICE):** Curvas amostra a amostra que desmascaram efeitos heterogêneos e interações entre atributos que o PDP médio esconde.
* **Feature Importances:** Mensuração baseada na redução de impureza das partições dos nós das árvores.

---
<a id="principais-conclusoes"></a>
## 📈 Principais Conclusões

* **Limitação Linear:** A regressão linear simples apresentou ajuste ruim, dependendo principalmente do termo constante para encontrar o valor predito.
* **Descritores-chave:** A magnetização total e a distância de ligação foram as variáveis mais decisivas nas quebras da árvore de decisão.
* **Comportamento Não Linear:** O centro da banda $d$ e a distância de ancoragem exibiram comportamentos marcadamente não monotônicos nas splines do GAM e nos degraus do PDP.
* **Efeito dos Sítios:** Os sítios do tipo *hollow* demonstraram maior dispersão nas trajetórias ICE em relação a *top* e *bridge*, indicando interação forte com a orientação da molécula adsorvida.

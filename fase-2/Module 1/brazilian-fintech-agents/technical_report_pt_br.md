# Relatório Técnico — Brazilian Fintech Agents

**Projeto:** FIAP — Fase 2, Módulo 1  
**Autores:** Equipe Brazilian Fintech Agents  
**Data:** 2026-09-13  
**Versão:** 2.0.0

---

## 1. Visão Geral

Este relatório descreve as decisões técnicas tomadas durante o design e a implementação do pipeline **Brazilian Fintech Agents** v2 — um sistema multiagente autônomo alimentado pelo **Google Gemini** que ingere dados de transações de cartão de crédito, compreende semanticamente o schema do dataset no contexto de um objetivo de análise definido pelo usuário, realiza análise exploratória de dados e gera um relatório executivo completamente redigido por LLM, sem intervenção humana.

O sistema processa o [dataset Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284.807 transações, 31 atributos, setembro de 2013), caracterizado por severo desbalanceamento de classes: transações fraudulentas representam apenas 0,172% do volume total.

**Princípio central de design:** O pipeline é **agnóstico ao dataset**. Ao receber um objetivo de análise explícito (`--goal`), ele pode analisar qualquer arquivo CSV sem conhecimento de domínio hardcoded.

---

## 2. Decisões de Arquitetura

### 2.1 Fluxo de Dados Sequencial em vez de Arquitetura Orientada a Eventos

**Decisão:** O pipeline adota um modelo linear e sequencial de fluxo de dados: cada agente executa até sua conclusão e repassa o resultado diretamente ao próximo agente via orquestrador.

**Justificativa:** A carga de trabalho é um processamento em lote (*batch*) — não há necessidade de paralelismo, filas de eventos ou processamento assíncrono. Um fluxo sequencial é:

- **Mais fácil de compreender** — cada etapa é determinística e inspecionável
- **Mais fácil de depurar** — falhas são imediatamente atribuídas a um agente específico
- **Suficiente para a escala** — o dataset cabe na memória (~150MB); processamento distribuído não é necessário

Uma arquitetura orientada a eventos ou assíncrona adicionaria complexidade sem nenhum benefício para um processamento batch em uma única máquina.

---

### 2.2 Agente 1 Orientado por Missão: Compreensão de Schema Guiada pelo Objetivo

**Decisão:** O `DataProfilerAgent` (Agente 1) recebe tanto o caminho do CSV quanto um **objetivo de análise** explícito (ex: `"detect fraudulent transactions"`). O prompt do LLM é construído em torno desse objetivo, fazendo o agente raciocinar sobre o dataset pela lente da missão.

**Justificativa:** Esta é a inovação central da v2. Em vez de hardcodar quais colunas são relevantes ou quais análises executar, o Agente 1 pergunta ao Gemini:

1. *O que cada campo representa?*
2. *Quais campos são relevantes para o objetivo e por quê?*
3. *Quais análises devem ser executadas para atingir o objetivo?*
4. *Quais sinais de fraude o próximo agente deve procurar?*

O resultado é um JSON `DataProfile` que contém `prescribed_analyses` — uma lista de etapas de análise com explicações explícitas de `goal_link`. **O Agente 2 lê essa lista e executa exatamente o que o Agente 1 prescreveu**, e não um conjunto fixo de funções hardcoded.

Isso significa que o mesmo pipeline pode analisar um dataset médico com objetivo `"detectar anomalias em pacientes"` ou um dataset logístico com objetivo `"identificar entregas atrasadas"` sem nenhuma alteração de código.

---

### 2.3 Separação de Responsabilidades em Três Agentes

**Decisão:** O pipeline é dividido em três agentes especializados — `DataProfilerAgent`, `AnalysisAgent` e `ReportWriterAgent` — cada um com uma única responsabilidade isolada.

**Justificativa:** Segue o **Princípio da Responsabilidade Única (SRP)**:

| Agente | Responsável por | Não responsável por |
|--------|----------------|---------------------|
| DataProfilerAgent | Compreensão do schema, alinhamento com o objetivo, prescrição de análises | Computação estatística, relatório |
| AnalysisAgent | Execução estatística (pandas), interpretação via LLM | I/O de dados, formatação |
| ReportWriterAgent | Redação do relatório (LLM), escrita de arquivo | Qualquer cálculo |

Benefícios:
- **Testabilidade independente** — cada agente pode ser testado unitariamente com entradas simuladas
- **Substituibilidade** — qualquer agente pode ser trocado sem afetar os demais
- **Clareza** — falhas em compreensão de schema, análise ou geração de relatório são imediatamente distinguíveis nos logs

---

### 2.4 Integração com LLM via Google Gemini

**Decisão:** Os três agentes utilizam o **Google Gemini** (`gemini-3.6-flash`) para suas tarefas de raciocínio. O Gemini é chamado via SDK `google-genai` com prompts estruturados em JSON.

**Justificativa:** O LLM é utilizado onde a lógica programática é insuficiente:

| Tarefa | Por que o LLM é necessário |
|--------|---------------------------|
| Descrição e atribuição de papel dos campos | Requer compreensão semântica de nomes de colunas e valores de amostra — inatingível apenas com inspeção de dtype |
| Prescrição de análises para qualquer dataset | Requer raciocínio sobre o objetivo em relação aos dados — não pode ser hardcoded sem conhecimento de domínio |
| Interpretação dos resultados estatísticos | "V17 tem r=-0,3135" requer explicação contextual |
| Redação do relatório executivo | Geração de linguagem natural com qualidade contextual não é possível com templates de string |

**Escolha do modelo — `gemini-3.6-flash`:**
- Inferência rápida (~20s por chamada) vs. `gemini-2.5-pro` (~60s)
- Qualidade de raciocínio suficiente para saída JSON estruturada e geração narrativa
- Custo-eficiente para jobs batch em produção

**Gerenciamento da chave de API:** Lida da variável de ambiente `GEMINI_API_KEY` (ou arquivo `.env` via `python-dotenv`). Nunca hardcoded. Nunca commitada no controle de versão.

---

### 2.5 Contratos de Dados Tipados via Dataclasses Python

**Decisão:** A comunicação entre agentes utiliza objetos `dataclass` Python tipados (`DataProfile`, `AnalysisResult`) em vez de dicionários brutos ou DataFrames.

**Justificativa:**

- **Contratos explícitos** — cada campo é nomeado e tipado; nenhum conhecimento implícito de chaves de dicionário é necessário
- **Suporte a IDE** — autocompleção e análise estática funcionam nativamente
- **Falha rápida** — campos ausentes ou com tipo incorreto geram erros no momento da construção
- **Sem overhead** — dataclasses são Python puro com custo de execução zero

**Contratos de dados principais:**

```python
@dataclass
class DataProfile:
    analysis_goal: str           # propagado para todos os agentes downstream
    target_field: str            # variável-alvo identificada pelo LLM
    fraud_signal_fields: list    # colunas de alto sinal identificadas pelo LLM
    prescribed_analyses: list    # o que o Agente 2 deve executar e por quê

@dataclass
class AnalysisResult:
    llm_interpretation: str        # narrativa do Gemini sobre os achados
    llm_actionable_insights: list  # 3 recomendações de negócio baseadas em dados
    feature_signals: list          # atributos mais correlacionados com explicações do LLM
```

---

### 2.6 Fallback Gracioso — Sem Dependência Rígida da Disponibilidade do LLM

**Decisão:** Cada chamada ao LLM tem um caminho de fallback. Se `GEMINI_API_KEY` estiver ausente ou a chamada de API falhar, cada agente continua com resultados apenas do pandas.

**Justificativa:**
- **Resiliência** — o pipeline nunca falha por causa de um erro de API
- **Testabilidade** — o pipeline pode ser testado de ponta a ponta sem uma chave de API
- **Separação de preocupações** — a correção computacional (pandas) é independente da qualidade narrativa (LLM)

O fallback é registrado como `[WARNING]` e o rodapé do relatório registra `source: fallback` vs. `source: llm`.

---

### 2.7 Ponto de Entrada via CLI com Objetivo Configurável

**Decisão:** `main.py` expõe os argumentos `--input`, `--output` e `--goal` via `argparse`.

**Justificativa:** O argumento `--goal` é o que torna o pipeline agnóstico ao dataset. Ele permite que o mesmo código seja aplicado a diferentes problemas analíticos alterando um único parâmetro de linha de comando.

---

## 3. Estratégia de Detecção de Anomalias

### 3.1 Detecção Estatística de Outliers: Método IQR no Valor das Transações

**Método** (executado pelo `AnalysisAgent` via pandas):

```
Q1 = Amount.quantile(0.25)  →  €5,60
Q3 = Amount.quantile(0.75)  →  €77,51
IQR = Q3 - Q1               →  €71,91
limite_superior = Q3 + 3 × IQR  →  €293,24

flagged = transações onde Amount > limite_superior
```

**Resultados no dataset:** 20 transações de alto valor sinalizadas acima do limite de €293,24 (máximo observado: €25.691,16).

---

### 3.2 Justificativa para o IQR em Relação às Alternativas

| Método | Considerado | Motivo da Não Adoção |
|--------|------------|----------------------|
| **Z-score** | Sim | Assume distribuição normal; valores de transação são fortemente assimétricos à direita |
| **Isolation Forest** | Sim | Requer treinamento de modelo ML; aumenta a complexidade sem ganho proporcional |
| **Clustering DBSCAN** | Sim | Alto custo computacional para 284K linhas; difícil de explicar para stakeholders não técnicos |
| **Limiar fixo** | Sim | Arbitrário; não é generalizável para datasets com faixas de valores diferentes |
| **IQR (escolhido)** ✅ | — | Agnóstico à distribuição; robusto à assimetria; sem treinamento de modelo; transparente; padrão em análise financeira |

**Por que 3× IQR e não 1,5× IQR:** O 1,5× IQR padrão sinalizaria ~7% de todas as transações como outliers em distribuição assimétrica à direita, gerando volume excessivo de falsos positivos. O multiplicador 3× produz um conjunto mais preciso de valores genuinamente extremos.

---

### 3.3 Detecção de Sinais de Fraude Interpretada pelo LLM

O `AnalysisAgent` computa a **correlação de Pearson** de cada atributo (V1–V28) com o rótulo `Class` via pandas e passa os resultados ao Gemini para interpretação contextual.

**O que o pandas computa:**

| Atributo | Correlação |
|----------|------------|
| V17 | −0,3135 |
| V14 | −0,2934 |
| V12 | −0,2507 |
| V10 | −0,2070 |

**O que o Gemini interpreta (do relatório gerado):**

> *"Vetor preditivo primário; desvios negativos pronunciados indicam risco severo de fraude. Priorizar V17 e V14 como atributos centrais em pipelines XGBoost/LightGBM de produção."*

Essa interpretação é **dinâmica** — muda se o dataset mudar, porque é gerada pelo LLM a partir dos valores reais de correlação, não pré-escrita.

---

### 3.4 Detecção de Anomalias Temporais

O pipeline agrupa a coluna `Time` em janelas horárias e computa a taxa de fraude por janela. O Agente 1 prescreve essa análise com um `goal_link` explícito:

> *"Identifica janelas de tempo onde o risco de fraude é elevado — suporta diretamente o objetivo de detectar transações fraudulentas."*

**Principais resultados:**

| Janela Horária | Taxa de Fraude | Multiplicador | Nível de Ameaça |
|----------------|---------------|---------------|-----------------|
| 26 | **1,556%** | 9,3× | CRÍTICO |
| 28 | **1,515%** | 9,1× | CRÍTICO |
| 2  | **1,335%** | 8,0× | CRÍTICO |

---

### 3.5 Por que Acurácia Padrão Não Foi Utilizada

Com apenas 0,167% das transações sendo fraudulentas, um classificador trivial que prevê "legítima" para toda transação atinge **99,83% de acurácia** sem detectar nenhuma fraude. O Gemini registra explicitamente esse fato no relatório executivo gerado e recomenda a **AUPRC (Área Sob a Curva Precisão-Revocação)** e balanceamento de classes via SMOTE para qualquer trabalho de ML subsequente.

---

## 4. Limitações e Trabalhos Futuros

| Limitação | Melhoria Sugerida |
|-----------|-------------------|
| O IQR detecta apenas outliers em `Amount`; outliers nos atributos V não são sinalizados diretamente | Aplicar distância de Mahalanobis ou Isolation Forest no espaço completo de atributos |
| Chamadas ao LLM adicionam ~75 segundos ao tempo de execução | Fazer cache do `DataProfile` para execuções repetidas no mesmo dataset |
| Sem ID de cliente no dataset; análise no nível do cliente não é possível | Solicitar dados desmascarados ao provedor mediante NDA |
| Pipeline apenas em batch; inadequado para prevenção de fraude em tempo real | Adicionar camada de streaming (ex.: Kafka + Faust) para pontuação online |
| Relatório é Markdown estático; sem visualizações interativas | Integrar com ferramenta de BI (ex.: Metabase, Grafana) ou adicionar exportações de gráficos |

---

## 5. Referências

- Dataset: [Credit Card Fraud Detection — Kaggle / ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Método IQR: Tukey, J.W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
- Recomendação AUPRC: Davis, J. & Goadrich, M. (2006). *The relationship between Precision-Recall and ROC curves*. ICML.
- Google Gemini SDK: [https://ai.google.dev/gemini-api/docs](https://ai.google.dev/gemini-api/docs)
- Arquitetura: [design/design.md](./design/design.md)
- Especificação: [spec/spec.md](./spec/spec.md)

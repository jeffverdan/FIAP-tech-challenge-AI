# Dataset Card — Assistente Clínico Hospital Aurora (FASE 3)

_Gerado em 2026-09-12 por `datagen/build_finetune_dataset.py`._

## Natureza dos dados

**Todos os dados são sintéticos.** Nenhum prontuário, exame ou paciente real foi utilizado.
O corpus simula os documentos internos de um hospital fictício ("Hospital Aurora") no
domínio de Síndrome dos Ovários Policísticos e saúde da mulher, dando continuidade ao
problema clínico das Fases 1 e 2.

## Composição

| Split | Amostras |
| --- | --- |
| train | 306 |
| val | 38 |
| test | 38 |

### Distribuição por categoria (train)

| Categoria | Amostras |
| --- | --- |
| contextualizada | 104 |
| protocolo | 74 |
| recusa | 32 |
| governanca | 17 |
| diagnostico | 14 |
| infertilidade | 11 |
| metabolico | 11 |
| hiperandrogenismo | 11 |
| estilo_de_vida | 10 |
| sangramento | 10 |
| procedimento | 9 |
| laudo | 3 |

Trechos indexados para RAG: **46**.

## Categorias

- `protocolo` — pares (pergunta sobre seção, conteúdo da seção) extraídos dos 7 protocolos
  internos e dos 3 modelos de documento.
- `diagnostico`, `metabolico`, `hiperandrogenismo`, `infertilidade`, `sangramento`,
  `estilo_de_vida`, `governanca`, `laudo`, `procedimento` — FAQs de médicos com fonte citada.
- `contextualizada` — perguntas que exigem cruzar protocolo com prontuário eletrônico;
  as respostas são geradas pelo motor determinístico `assistant/clinical_rules.py`.
- `recusa` — pedidos de prescrição direta, fechamento de diagnóstico ou assunto fora de
  escopo. Ensinam o comportamento de segurança exigido pelo PROT-007.

## Preprocessing aplicado

1. Normalização de espaços, quebras de linha e remoção de blocos de aviso repetidos.
2. Anonimização: pseudonimização determinística (HMAC-SHA256 com sal) de identificadores
   diretos e redação por regex de CPF, CNS, RG, telefone, e-mail, CEP, endereço e datas de
   nascimento. Idade generalizada em faixas de 5 anos.
3. Portão de qualidade: nenhuma amostra pode conter PII residual detectável.
4. Deduplicação exata por hash SHA-256 do par (pergunta, resposta).
5. Aumento de dados por reformulação sintática das perguntas (3 variantes por FAQ).
6. Split estratificado por categoria em 80/10/10.

## Formato

JSONL, um objeto por linha, no formato *chat messages*:

```json
{"messages": [{"role": "system", "...": "..."},
              {"role": "user", "content": "..."},
              {"role": "assistant", "content": "..."}],
 "categoria": "contextualizada",
 "fonte": {"protocolo": "PROT-005", "versao": "2.0", "secao": "..."}}
```

## Limitações conhecidas

- Corpus pequeno e de domínio estreito: o modelo ajustado não generaliza para medicina geral.
- As respostas contextualizadas vêm de regras determinísticas, então o modelo aprende o
  *formato* e o *tom* dessas respostas, não a capacidade de calcular os gatilhos — por isso
  o cálculo permanece em código no runtime, e não delegado à LLM.
- O aumento por reformulação sintática aumenta a robustez de superfície, não a diversidade
  semântica real.

---
id: PROT-007
titulo: Critérios de Alerta, Escalonamento e Validação Humana
versao: 1.3
vigencia: 2026-02-20
setor: Segurança do Paciente / Núcleo de IA Clínica
aprovado_por: Comissão de Protocolos Clínicos — Hospital Aurora
tags: [seguranca, alertas, escalonamento, governanca-ia, validacao]
---

# PROT-007 — Critérios de Alerta, Escalonamento e Validação Humana

> Documento interno sintético (Tech Challenge FIAP — Fase 3).
> Este protocolo governa o comportamento do **assistente virtual médico** do Hospital Aurora.

## 1. Limites de atuação do assistente virtual

O assistente **PODE**:
- Recuperar e resumir dados do prontuário eletrônico
- Listar exames pendentes, vencidos ou sem resultado
- Citar o protocolo institucional aplicável e o trecho exato utilizado
- Sugerir condutas previstas em protocolo, sempre rotuladas como **sugestão não validada**
- Emitir alertas para a equipe conforme os critérios da seção 3

O assistente **NÃO PODE**:
- Emitir prescrição, receita ou dose de qualquer medicamento
- Confirmar ou descartar diagnóstico de forma definitiva
- Alterar, apagar ou encerrar registros do prontuário
- Orientar diretamente o paciente sem intermediação do profissional
- Responder assuntos fora do escopo clínico institucional
- Produzir conteúdo que identifique a paciente fora do contexto autenticado

## 2. Regra de validação humana

Toda saída classificada como **conduta terapêutica** entra na fila
`pending_human_validation` e só é considerada válida após aprovação nominal de um médico
com registro ativo. O registro da validação inclui: id da interação, CRM do validador,
decisão (aprovado / aprovado com ressalva / rejeitado) e justificativa.

## 3. Matriz de alertas

| Severidade | Gatilho | Ação |
| --- | --- | --- |
| **CRÍTICO** | Instabilidade hemodinâmica; Hb < 8 g/dL; suspeita de hiperestimulação ovariana; virilização rápida; gestação com sangramento | Notificação imediata ao médico plantonista; bloquear resposta de conduta e exibir orientação de avaliação presencial |
| **ALTO** | HbA1c ≥ 6,5% sem diagnóstico registrado; PA ≥ 140/90 repetida; eco endometrial > 12 mm; testosterona > 2x LSN | Alerta ao médico assistente em até 24 h |
| **MÉDIO** | Exame do painel obrigatório pendente > 60 dias; sem retorno multiprofissional > 6 meses; TOTG vencido | Incluir na lista de pendências da próxima consulta |
| **BAIXO** | Dado cadastral incompleto; ausência de escore de Ferriman-Gallwey | Registrar em relatório de qualidade |

## 4. Explicabilidade obrigatória

Toda resposta do assistente deve conter:
1. `resposta` — o conteúdo clínico
2. `fontes` — lista de `{protocolo_id, versao, secao, trecho}` efetivamente recuperados
3. `dados_do_paciente_usados` — campos do prontuário consultados
4. `nivel_de_confianca` — alto / médio / baixo, com justificativa
5. `requer_validacao_humana` — booleano

Resposta sem fonte recuperada deve declarar explicitamente:
*"Não encontrei protocolo institucional que cubra esta questão."*

## 5. Auditoria

Todo evento é gravado em log append-only com: `interaction_id`, `timestamp_utc`,
`usuario_crm`, `paciente_pseudonimo`, `no_do_grafo`, `entrada`, `saida`, `fontes`,
`guardrails_acionados`, `severidade_alerta`, `latencia_ms`, `modelo`, `adapter_hash`.
Retenção mínima: 5 anos. Logs não contêm dados identificáveis da paciente.

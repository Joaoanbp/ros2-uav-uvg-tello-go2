# Avaliação reproduzível da profundidade RGB

O conjunto em `scripts/perception_scenarios.py` gera oito cenários determinísticos:
área externa vazia, sala vazia, parede, árvore, postes de 8 cm, pouca textura,
pouca luz e luz mais forte. A sala é uma aproximação procedural, não uma cena
fotorealista. Cada cenário tem três posições fixas de câmera, a 1,6 m de altura.
Alterações de luz usam a mesma geometria da sala vazia. A semente das texturas
é 2026.

O coletor posiciona o drone como uma bancada de teste. **Não é um voo autônomo**.
Salva RGB e profundidade Gazebo pareados em até 5 ms, pose em até 20 ms, calibração
e configuração do mundo. O replay usa exclusivamente RGB no modelo, sem ajuste
de escala baseado na referência. Os mesmos quadros podem ser reavaliados sem
abrir o Gazebo.

## Reproduzir

Com o código/pesos instalados conforme [o guia RGB](rgb-depth-experiment.md),
execute dentro do contêiner robotics:

```bash
PYTHONPATH=/workspace/scripts python3 /workspace/scripts/evaluate_rgb_depth.py --capture
```

São 24 vistas, avaliadas em entradas de 252 e 518 pixels, com duas inferências
medidas por vista. O aquecimento é registrado separadamente. O relatório salva
o hash do checkpoint e o commit do código do modelo. Para repetir apenas a
inferência, omita `--capture`. Para acrescentar o diagnóstico de campo de visão:

```bash
PYTHONPATH=/workspace/scripts python3 /workspace/scripts/evaluate_rgb_depth.py --diagnose-fov
```

Para recalcular as métricas e o HTML a partir dos resultados salvos, sem PyTorch:

```bash
PYTHONPATH=scripts python3 scripts/evaluate_rgb_depth.py --refresh-report
```

O resultado fica em `artifacts/perception-evaluation/report.html`, com dados
completos em `report.json`, quadros em `dataset/` e painéis em `results/`.

## O que é medido

- Erro absoluto e sinal do erro, intervalos de distância e pior p95 entre vistas.
- Pixels de referência abaixo de 4 m cuja distância é superestimada em mais de
  0,5 m. O relatório geral inclui piso; a coluna **flight band** considera apenas
  superfícies próximas à altura de voo (±0,375 m). Sem superfície nessa faixa,
  o resultado é `n/a`, não sucesso de detecção.
- Falsos pixels próximos: estimativa abaixo de 2 m quando a referência é ≥2,5 m.
- Predições inválidas, que contam como medidas perdidas, qualidade da imagem,
  voxels estimados e motivos de rejeição pelas heurísticas atuais.
- Atraso de inferência CPU. Esse replay não mede toda a latência de navegação;
  a captura ao término da inferência também é registrada nas missões ao vivo.

Os limites exploratórios são MAE médio das vistas ≤0,30 m, p95 de cada vista
≤0,50 m, superestimações próximas ≤1%, perda na faixa de voo ≤1%, p95 de inferência
≤0,75 s e nenhuma rejeição heurística. Não constituem certificação de segurança.

## Resultado desta execução (2026-10-16)

Nenhuma combinação de cenário e resolução passou todos os critérios.

| Cenário | MAE 252 px | MAE 518 px | Perda na faixa de voo 252 / 518 px |
| --- | --- | --- | --- |
| Área externa vazia | 1,65 m | 2,33 m | n/a |
| Sala vazia | 0,93 m | 0,47 m | n/a |
| Parede | 0,81 m | 0,28 m | 100% / 0,20% |
| Árvore | 0,36 m | 0,96 m | 10,03% / 0,08% |
| Postes finos | 1,29 m | 0,71 m | 100% / 1,11% |
| Pouca textura | 1,38 m | 2,22 m | n/a |
| Pouca luz | 1,35 m | 2,05 m | n/a |
| Luz mais forte | 1,06 m | 0,48 m | n/a |

O p95 da inferência ficou em 0,39–0,44 s a 252 px e 3,38–4,00 s a 518 px neste
CPU, usando quatro threads. A resolução maior ajuda na parede e nos postes,
mas não resolve todos os cenários e excede o prazo de navegação.

A parede de posição conhecida e o piso confirmaram a projeção axial e os
extrínsecos com resíduos de aproximadamente 2–3 mm. O wrapper RGB e a inferência
oficial diferiram em menos de 0,00003 m no quadro comparado. Portanto os grandes
erros não são explicados pela troca RGB/BGR ou pela projeção.

A câmera simulada tem HFOV de 80°; o treinamento Hypersim usa aproximadamente
60°. Comparando os mesmos pixels centrais, um recorte de 60° reduziu o MAE de
0,90 para 0,24 m na sala e de 0,82 para 0,17 m na parede. Piorou a árvore
(0,47 → 0,87 m), a área externa (2,21 → 2,75 m) e a pouca textura
(1,64 → 2,83 m). Esse recorte é somente diagnóstico e deixaria bordas sem
observação; não foi ativado na navegação.

As heurísticas de iluminação e crescimento de voxels aceitaram algumas vistas
com erros perigosos, incluindo postes. Por isso `MetricIndoorDepth.navigation_ready`
permanece falso: essa fonte registra observações, mas não autoriza avanço
horizontal, mesmo que as heurísticas passem. Quadros rejeitados por qualidade ou
crescimento excessivo de superfícies não são integrados ao mapa. A linha de base Gazebo permanece disponível explicitamente
com `--depth-source gazebo`.

O próximo trabalho é validar uma estimativa que resolva os erros de escala,
textura e estruturas finas no conjunto fixado, preservando também o prazo de
processamento. A pose por câmera continua sendo uma etapa independente; este
ensaio usa pose do simulador e não mede drift de localização.

## Demonstração ao vivo

O ensaio gráfico de comparação percorreu todo o perímetro circular de 2 m,
mas parou durante a varredura ao observar superfícies que invadiram a margem
inflada na posição atual. O teste completo de missão, portanto, não passou;
seu gráfico e log estão em `artifacts/circle-camera-flight-check.*`. Esse
resultado não invalida o teste pontual anterior de ida/desvio/volta, mas limita
o que se pode afirmar sobre missões completas.

Na primeira execução RGB com Gazebo e painel abertos, a inferência levou
0,75 s e o atraso desde o recebimento chegou a 0,79 s. O limite de 0,75 s
interrompeu o avanço. O conjunto de replay sem essas janelas não mede essa
carga adicional. O painel identifica a fonte utilizada e mantém o motivo
da parada visível; fechar o gráfico encerra o Gazebo daquela sessão.

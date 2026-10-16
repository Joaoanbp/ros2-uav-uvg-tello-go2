# Navegação RGB com YOLO e geometria conhecida

O padrão de `./scripts/zigzag.sh` agora usa `--depth-source yolo`.
O lançador gera `artifacts/tello-yolo-rgb.sdf`: câmeras frontal e inferior do
tipo **camera**, sem `rgbd_camera`, `depth_camera` ou tópico de profundidade.
O mapa começa vazio. A navegação recebe somente RGB, intrínsecos e a pose
do próprio drone. O detector não lê cenas, poses de obstáculos ou o conjunto
de treinamento durante o voo.

## Executar

No ambiente já preparado:

```bash
./scripts/zigzag.sh --shape circle --radius 3
```

Equivalente explícito, com registro:

```bash
./scripts/zigzag.sh --shape circle --radius 3 --depth-source yolo \
  --evaluation-dir /workspace/artifacts/yolo-live
```

O raio é dado em metros. Base, altitude, cobertura, prévia e escolha da entrada
continuam usando os planejadores compartilhados. A entrada no perímetro é a
projeção mais próxima da base. O painel mostra caixas da última observação YOLO,
ocupação estimada, rota A* e trajetória executada.

O lançador também coloca uma caixa azul de 20 × 20 × 15 cm no chão, em uma
posição aleatória dentro da ROI, sem sobrepor o pilar nem a base. A posição
muda a cada lançamento, independentemente da semente do polígono. A caixa é
somente um alvo de cena: sua identificação, estimativa visual de posição e
envio ao Go2 ainda não foram implementados. O fixture de treinamento não inclui
essa caixa.

## Cobertura adaptativa

O padrão é `--coverage-mode adaptive`. O perímetro mantém a entrada mais próxima
da base e a volta completa. Depois disso as faixas são guias, e o navegador
escolhe trechos que ainda acrescentam observação do solo. Pode contornar um
objeto e seguir em outra faixa, inclusive saltando da faixa 10 para a 12.
Partes não vistas da faixa 11 continuam pendentes e podem receber uma passagem
de acabamento depois. O próximo trecho considera ganho de cobertura, distância,
obstáculos e continuidade; não há regra de saltar sempre um número fixo.

A grade usa células de 20 cm dentro da região. Uma célula só recebe crédito
quando seus quatro cantos estão dentro de uma imagem inferior RGB recente,
calibrada e sincronizada com a pose medida, respeitando também o diâmetro de
footprint solicitado. O campo de visão real limita esse footprint. Previsões
de imagens futuras servem apenas para escolher uma rota, sem contar como
observação. Oclusões pelo pilar permanecem pendentes. Quando um novo objeto é
mapeado, as vistas anteriores são reavaliadas para retirar crédito de regiões
que ele ocultava.

A envoltória estimada do objeto **incluindo a incerteza** é contabilizada
separadamente como excluída, nunca como solo observado. A margem de voo de
50 cm não é excluída da cobertura. A conclusão exige zero células pendentes
fora da envoltória; se não houver vista acessível, a missão para como incompleta.
Essa medida é geométrica, na resolução da grade e sob a hipótese de chão plano,
não uma certificação de que um alvo foi reconhecido em cada imagem.

As ligações entre faixas usam A* sem obrigação de retornar à faixa anterior.
Curvas quadráticas arredondam os cantos quando todos os segmentos resultantes
respeitam a ocupação inflada e a região da missão. Quando não cabe uma curva
segura, o canto é mantido. Cada passo continua exigindo percepção frontal nova.
O mapa mostra solo observado em verde, pendente em amarelo e a envoltória
excluída em cinza. O registro inclui os RGBs inferiores, poses, calibração,
ganho de cobertura e escolhas de faixa.

`--side-overlap` e `--footprint-width` ainda geram as mesmas guias nominais da
prévia e da exportação. O modo adaptativo reduz a repetição dessas guias e
**não exige repetir a sobreposição nominal em cada passagem real**. Para
missões que precisam dessa repetição, use a execução anterior:

```bash
./scripts/zigzag.sh --shape circle --radius 3 --coverage-mode fixed
```

## Domínio deste protótipo

O objeto é **um pilar estático laranja/preto, vertical, de 0,8 m de diâmetro e
1,8 m de altura**, apoiado no chão plano `z=0`. Essas dimensões são uma hipótese
explícita, autorizada para este teste controlado. Sua localização não é dada
ao navegador. A pose do drone ainda é a do simulador. A base e a subida vertical
precisam estar livres, como nas missões anteriores.

Um YOLOv8n treinado para `calibration_pillar` fornece a caixa na imagem. O código
em `scripts/yolo_geometry.py` ajusta a projeção de um cilindro conhecido a essa
caixa e calcula posição XY, erro de reprojeção e incerteza. A ocupação inclui
todo o volume conhecido e a incerteza; a grade acrescenta a margem de 50 cm.
O A* e a execução em passos de até 18 cm são os existentes. O detector usa NMS
com IoU de 0,5 para suprimir caixas duplicadas.

A primeira estimativa consistente ancora a envoltória conservadora do objeto
estático. Novas medidas verificam sua consistência, evitando que ruído de caixas
desloque o mapa durante um desvio. A ocupação permanece quando o objeto sai do
campo de visão. Uma detecção central sem contato com o chão ou sem ajuste
confiável interrompe a missão. Uma vista correspondente ao objeto já mapeado,
inclusive parcialmente cortada, pode reutilizar a envoltória estática anterior
somente quando a caixa visível corresponde à projeção do objeto já mapeado
(IoU ≥0,5); não se inventa uma distância nova a partir dessa vista.
Caixas duplicadas associadas ao mesmo objeto formam um único track; detecções
centrais sem associação ou geometria confiável continuam causando parada.
Imagem/pose antiga, imagem escura e mudança
incompatível de posição também interrompem o avanço.

Isso não é um detector geral de árvores, fios, paredes, pessoas ou objetos
móveis. Ausência de detecção não certifica espaço livre fora do domínio do
teste. O Tello físico ainda exige localização e calibração validadas em conjunto.

## Preparar ou reproduzir o treinamento

Os pesos locais ficam em `models/yolo_pillar.pt`, com perfil e hash em
`models/yolo_pillar.json`; esses arquivos gerados são ignorados pelo Git.
O perfil só autoriza movimento após a validação separada passar, e seu hash
deve corresponder aos pesos carregados.

Dependências testadas dentro do contêiner, além do ambiente Gazebo:

```bash
python3 -m pip install --user numpy==1.26.4 opencv-python==4.10.0.84 scipy==1.13.1
python3 -m pip install --user torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install --user ultralytics==8.3.40
mkdir -p /workspace/models
curl -fL -o /workspace/models/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
PYTHONPATH=/workspace/scripts python3 /workspace/scripts/prepare_yolo_pillar.py --capture --train --validate
```

Foram usadas 180 imagens de treinamento, 40 de validação e 48 de teste, com
sementes independentes, posições/ângulos/distâncias variadas e imagens negativas.
As caixas são rotuladas pela geometria **somente na preparação offline**.
O treinamento executa 30 épocas de ajuste de um detector pequeno. O conjunto,
manifesto e métricas ficam em `artifacts/yolo-pillar/`.

A validação mede posição contra a geometria do fixture, sem usar profundidade.
Nesta execução: 42 vistas elegíveis detectadas, nenhuma perda, nenhum falso
positivo nas seis vistas negativas, erro médio de posição 4,5 cm e máximo de
16,7 cm; todos ficaram dentro da incerteza estimada. Isso é resultado desse
conjunto controlado, não uma garantia de desempenho geral.

API e treinamento: [detecção](https://docs.ultralytics.com/tasks/detect/),
[treinamento Ultralytics](https://docs.ultralytics.com/modes/train/).

## Verificação completa no Gazebo

```bash
PYTHONPATH=/workspace/scripts python3 /workspace/tests/check_yolo_navigation_gazebo.py
PYTHONPATH=/workspace/scripts python3 /workspace/tests/check_yolo_navigation_gazebo.py --adaptive
```

O teste muda nome e posição do pilar sem passá-los à percepção. Verifica a
ausência do sensor/tópico de profundidade e de sua assinatura, mapa inicial
vazio, perímetro com entrada próxima, desvio A*, nenhum waypoint pulado,
retorno, pouso, distância mínima medida e parada quando as imagens cessam.
Resultados em `artifacts/yolo-navigation/result.json` e `result.png`.

Nesta execução o teste completo passou, com dois desvios A*, nenhum waypoint
pulado e distância mínima medida de 1,45 m ao centro / 1,05 m à superfície do
pilar. A perda de imagens também impediu nova observação de movimento. As únicas
assinaturas da percepção foram imagem frontal RGB, calibração e pose do drone.

Antes da cobertura adaptativa, foi executado o lançador gráfico com raio de
3 m e as 15 faixas rígidas: perímetro completo, dez desvios A*, nenhum waypoint pulado, retorno à
base `(-7, -5)` e pouso em `z=0,09 m`. Foram registradas 779 decisões, nenhuma
parada e zero quadros de profundidade de referência; o percentil 95 do
processamento foi de 68 ms. O gráfico final fica em
`artifacts/yolo-live/fixed-result.png` e o resumo em
`artifacts/yolo-live/summary.json`.

O teste adaptativo separado com nome e posição diferentes do pilar passou:
sete trechos úteis em vez das 15 faixas nominais, cinco desvios A*, zero
waypoints inacessíveis, retorno e pouso. As 562 células requeridas foram vistas;
154 células da envoltória incerta foram reportadas como excluídas, com zero
células pendentes. A distância mínima medida foi de 1,57 m ao centro / 1,17 m
à superfície real do pilar. A perda de RGB frontal e inferior foi recusada.
Resultados em `artifacts/yolo-adaptive/result.json` e `result.png`.

O lançador gráfico também passou com o pilar padrão, a mesma base e raio de
3 m usados na comparação anterior:

| Execução | Trechos de varredura | Desvios A* | Distância horizontal comandada |
| --- | ---: | ---: | ---: |
| Faixas rígidas | 15 | 10 | 128,17 m |
| Cobertura adaptativa | 8 | 3 | 48,31 m |

As distâncias incluem perímetro e ida/retorno à base; são comandos da simulação
cinemática, não medidas de autonomia da bateria. A sequência adaptativa foi
4 → 8 → 12 → 13 → 10 → 6 → 4 → 3, incluindo acabamento de partes não vistas.
Foram observadas as 560 células requeridas e excluídas separadamente 156 da
envoltória incerta, sem pendências, waypoints inacessíveis ou paradas. O drone
retornou a `(-7, -5)` e pousou em `z=0,09 m`. Nenhum quadro de profundidade foi
registrado. O resumo completo está em `artifacts/yolo-adaptive-live/summary.json`,
e o gráfico do voo em `artifacts/zigzag-circle-voo.png`.

A suíte com 71 testes passou dentro do contêiner, incluindo crédito por imagem,
oclusão, parada com imagem inferior antiga, salto de faixa, curvas com margem
e contenção em regiões côncavas.

`--depth-source gazebo` mantém a comparação antiga com árvore e RGB-D.
`--depth-source rgb` mantém o experimento denso que reprovou a avaliação;
ele continua impedido de autorizar avanço. O modo YOLO não usa esses caminhos.

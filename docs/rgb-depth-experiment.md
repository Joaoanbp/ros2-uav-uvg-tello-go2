# Experimento RGB → profundidade métrica

Com `--depth-source rgb`, o voo cinemático de `zigzag_roi.py` e `drone_zigzag.py` usa a imagem
RGB frontal e o checkpoint **Depth Anything V2 Metric Hypersim Small**. O padrão
atual é [YOLO com geometria conhecida](yolo-rgb-navigation.md). A
profundidade sintética do Gazebo só entra no caminho de navegação quando
`--depth-source gazebo` é escolhido explicitamente. Com `--evaluation-dir`,
ela é assinada separadamente e salva para comparação; não é entrada do modelo,
do mapa, do A* ou da decisão de avanço.

**Estado atual:** o conjunto controlado reprovou essa fonte. O avanço horizontal
RGB fica bloqueado por `navigation_ready=False`; a simulação registra a
observação e mantém posição. Veja [método, resultados e reprodução](perception-evaluation.md).

## Preparação offboard

No ambiente Python que executa a missão, instale PyTorch, OpenCV e as
dependências do repositório oficial. Obtenha o código e o checkpoint nos links
do [README oficial de profundidade métrica](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/metric_depth/README.md):

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2 third_party/Depth-Anything-V2
python3 -m pip install --user 'numpy==1.26.4' 'opencv-python==4.10.0.84'
python3 -m pip install --user 'torch==2.2.2' 'torchvision==0.17.2' --index-url https://download.pytorch.org/whl/cpu
mkdir -p models
# Salve o checkpoint Hypersim Small como:
# models/depth_anything_v2_metric_hypersim_vits.pth
```

Esses arquivos grandes são ignorados pelo Git. No contêiner, instale os pacotes
no ambiente Python usado pela missão; o Dockerfile atual não instala PyTorch.
Execute uma missão de perímetro e registre os dados:

```bash
./scripts/zigzag.sh --shape circle --depth-source rgb \
  --depth-repository /workspace/third_party/Depth-Anything-V2 \
  --depth-checkpoint /workspace/models/depth_anything_v2_metric_hypersim_vits.pth \
  --evaluation-dir /workspace/artifacts/rgb-depth/circle
```

Para a linha de base, repita em outro diretório com `--depth-source gazebo`.
O relatório RGB é gerado por:

```bash
PYTHONPATH=scripts python3 scripts/perception_eval.py artifacts/rgb-depth/circle
```

O teste de integração curto também aceita `--rgb`:

```bash
PYTHONPATH=/workspace/scripts python3 /workspace/tests/check_depth_navigation_gazebo.py --rgb
```

Cada `frame-*.npz` contém RGB, estimativa e referência sincronizada. O JSONL
contém pose, carimbo de tempo, duração da inferência, qualidade simples da
imagem, comandos de avanço e paradas. O relatório calcula erro absoluto em
metros e pixels próximos cujo erro superestima a distância em mais de 0,5 m.
Separe execuções por cenário (rota limpa, árvore, pouca textura, iluminação
alterada). Frames sem referência sincronizada ficam marcados e não entram no
erro. Isso é avaliação de pixels, não prova de distância segura em todos os
obstáculos.

Uma imagem atrasada, incompleta, muito escura, saturada ou quase uniforme causa
parada. Essas heurísticas não certificam a confiança da estimativa. A pose ainda vem do Gazebo. **A localização sem pads e seu drift não
foram implementados ou validados**, portanto o relatório marca esse campo como
indisponível. O modelo pode errar com aparência plausível; nenhuma missão física
autônoma deve usar este experimento antes de medir conjuntamente erro de
distância, obstáculos perdidos, drift de localização e atraso em dados reais.

O padrão usa entrada de 252 pixels e quatro threads CPU para tentar atender ao
limite de 0,75 s por observação; `--depth-input-size` e `--depth-cpu-threads`
permitem medir outras configurações. Reduzir a entrada pode perder obstáculos
finos. Uma medição local no contêiner levou cerca de 0,4–0,7 s com 252–308
pixels e quatro threads; isso não substitui a medição durante o voo.
## Próximas etapas de validação

1. Executar cenários reproduzíveis no Gazebo e revisar trajetórias e relatório.
2. Avaliar localização por câmera sem pads em uma execução separada, comparando
   a pose estimada à pose do simulador; depois executar as duas estimativas juntas.
3. Só então ligar vídeo SDK e telemetria do Tello EDU em computador offboard,
   começando com voo curto, supervisionado, usando mission pads.
4. Após o mapa funcionar, testar um marcador ArUco ou objeto específico como
   alvo de interesse e depois trocar por detector adequado à busca e salvamento.
5. Para o Go2, transmitir posição do alvo em quadro de coordenadas compartilhado,
   incerteza e evidência visual. O Go2 faz sua própria localização e rota no solo.

Esta fonte de profundidade densa ainda não tem voo aprovado. O modo separado
[YOLO com geometria conhecida](yolo-rgb-navigation.md) passou em uma simulação
controlada com pilar. Estimativa de pose por câmera, conexão SDK física e
integração Go2 continuam pendentes.

### Resultado inicial observado

Em `tests/check_depth_navigation_gazebo.py --rgb`, a primeira vista, com a árvore
atrás da câmera, gerou 820 voxels estimados. O guardião parou antes do primeiro
avanço. No quadro sincronizado, o erro absoluto médio foi 1,49 m e o percentil
95 foi 5,74 m; 2.267 de 134.400 pixels de referência distante foram previstos
como próximos. Esses números são de um quadro e **reprovam** a condição de
entrada da missão, não representam desempenho agregado. O teste com
`--depth-source gazebo` completou ida, desvio, volta e parada por sensor ausente.

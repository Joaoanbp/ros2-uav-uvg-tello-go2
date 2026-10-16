# Detecção de obstáculos para UAVs: algoritmos e evidências experimentais

Consulta das fontes: **9 de outubro de 2026**. Este levantamento reúne **seis
artigos com experimentos em veículos reais** e relaciona seus módulos de
percepção ao projeto Tello/Gazebo. Não é uma revisão sistemática nem um ranking:
sensores, cenários, velocidades e métricas diferem entre os trabalhos.

## Estado do projeto e significado do mapa atual

**Contexto histórico:** esta seção descreve a linha de base RGB-D de outubro
de 2026. O padrão atual usa [YOLO, geometria conhecida e cobertura adaptativa
com câmeras somente RGB](../docs/yolo-rgb-navigation.md). O levantamento dos
artigos abaixo permanece uma referência de pesquisa, não a descrição do modo
padrão atual.

O voo e o painel usam `scripts/depth_navigation.py`: profundidade frontal RGB-D,
reprojeção métrica, remoção do solo, voxels de superfícies observadas e A* com
inflação de obstáculos na altura do drone. O mapa começa vazio e recebe somente
medidas visíveis. A pose do próprio drone vem do simulador para localização;
nenhuma geometria ou pose dos obstáculos é fornecida à percepção. O módulo
antigo `scripts/obstacle_map.py` permanece apenas para avaliação de referência.

A câmera frontal agora é **RGB-D sintética**; a inferior continua RGB. Elas
**não constituem um par estéreo retificado**. A profundidade é produzida por um
[sensor do Gazebo](https://gazebosim.org/api/sensors/8/classgz_1_1sensors_1_1RgbdCameraSensor.html),
não por uma rede monocular. A implementação geométrica é uma base para cena
estática, não uma reprodução completa de [3] ou [4]: não inclui rastreamento
dinâmico, Kalman, previsão de movimento ou planejamento probabilístico. A
simulação continua cinemática, com altitude padrão de 1,6 m.

Neste documento, **detecção** identifica um obstáculo ou risco de colisão;
**rastreamento** associa observações no tempo; **mapeamento** posiciona geometria
em um referencial; **desvio** escolhe comandos ou trajetórias. Um artigo pode
avaliar o sistema completo, sem provar isoladamente cada módulo.

## Mapa dos métodos

| Família / referência | Entrada | Saída da percepção | Adequação ao projeto — avaliação deste levantamento |
| --- | --- | --- | --- |
| Expansão aparente + SURF/template matching [1] | Sequência monocular | Aproximação de obstáculo frontal | Usa RGB; depende do movimento e de detalhes visuais. Não entrega diretamente a envoltória no mundo. |
| CNN DroNet [2] | Imagem monocular | Probabilidade de colisão e direção sugerida | Pode fornecer alerta; exige adaptação e validação do domínio. Não fornece caixas 3D. |
| Pares de profundidade + rastreamento [3] | Câmera estéreo e estado do UAV | Posição, velocidade e tamanho | Há profundidade simulada; associação temporal e previsão ainda não implementadas. |
| Propostas por profundidade/voxels + Kalman [4] | RGB-D, mapa e estado | Obstáculos dinâmicos rastreados e previsão | Referência para evoluir o mapa estático de voxels para rastreamento dinâmico. |
| Divergência do fluxo óptico + EKF [5] | Fluxo óptico e informação inercial | Risco, distância próxima e desaceleração | Alternativa leve para alerta frontal; exige sincronização e estimativa de movimento. |
| Segmentação temporal de eventos [6] | Câmera de eventos | Obstáculos em movimento e reação rápida | Requer outro sensor; não funciona diretamente nos quadros RGB atuais. |

As recomendações da última coluna são **inferências para este repositório**;
nenhum dos seis artigos testou este modelo simulado do Tello.

## 1. Expansão aparente de obstáculos frontais — Mori e Scherer, 2013

**Artigo:** *First Results in Detecting and Avoiding Frontal Obstacles from a
Monocular Camera for Micro Unmanned Aerial Vehicles*.
**Autores:** Tomoyuki Mori e Sebastian Scherer.
**Publicação:** ICRA 2013, pp. 1750–1757.

- **Algoritmo:** correspondências SURF e comparação de templates para estimar
  crescimento aparente de regiões da imagem, inclusive na direção frontal em
  que o fluxo óptico translacional pode ser pequeno.
- **Validação:** os autores relatam **20 experimentos de voo bem-sucedidos** com
  um pequeno quadrotor; o artigo inclui experimentos de detecção e desvio.
- **Limite para adoção:** resultado em configurações testadas, sem garantia para
  vegetação pouco texturizada ou qualquer arranjo de obstáculos. Não produz um
  mapa métrico completo por si só.

Fontes primárias conferidas: [registro da CMU](https://publications.ri.cmu.edu/first-results-in-detecting-and-avoiding-frontal-obstacles-from-a-monocular-camera-for-micro-unmanned-aerial-vehicles)
e [artigo completo](https://www.ri.cmu.edu/pub_files/2013/5/monocularObstacleAvoidance.pdf).

## 2. Predição monocular de colisão — DroNet, 2018

**Artigo:** *DroNet: Learning to Fly by Driving*.
**Autores:** Antonio Loquercio, Ana I. Maqueda, Carlos R. del-Blanco e Davide Scaramuzza.
**Publicação:** IEEE Robotics and Automation Letters, 2018.

- **Algoritmo:** CNN residual de oito camadas que estima direção e probabilidade
  de colisão a partir de uma imagem.
- **Validação:** a página dos autores documenta voos reais em cenários urbanos
  e interiores, incluindo corredores e estacionamentos, após treinamento com
  dados de carros e bicicletas. Há código, pesos e dados de colisão disponíveis.
- **Limite para adoção:** a saída é uma avaliação de risco e direção, não a
  posição/tamanho de cada obstáculo. Mudança para o visual sintético e a câmera
  inclinada deste projeto precisa de validação própria.

Fonte primária conferida: [página do projeto, artigo, código e dados](https://rpg.ifi.uzh.ch/dronet.html).

## 3. Detecção e rastreamento com estéreo — Lin et al., 2020

**Artigo:** *Robust Vision-based Obstacle Avoidance for Micro Aerial Vehicles in
Dynamic Environments*.
**Autores:** Jiahao Lin, Hai Zhu e Javier Alonso-Mora.
**Publicação:** ICRA 2020; preprint arXiv:2002.04920.

- **Algoritmo:** detecção e rastreamento a partir de pares de imagens de
  profundidade, estimando posição, velocidade e dimensão dos obstáculos.
  O sistema usa essas estimativas em um controlador preditivo com restrições
  probabilísticas de colisão (CC-MPC).
- **Validação:** implementação embarcada em quadrotor com câmera estéreo,
  com testes de desvio de obstáculos móveis em diferentes ambientes.
- **Limite para adoção:** o CC-MPC é a etapa de controle, não o detector.
  A integração demanda profundidade métrica, estado do drone e modelagem de
  incerteza; as duas câmeras RGB atuais não substituem esse sensor estéreo.

Fonte primária conferida: [resumo e artigo dos autores no arXiv](https://arxiv.org/abs/2002.04920).

## 4. Mapa e rastreamento de obstáculos dinâmicos RGB-D — Xu et al., 2023

**Artigo:** *A real-time dynamic obstacle tracking and mapping system for UAV
navigation and collision avoidance with an RGB-D camera*.
**Autores:** Zhefan Xu, Xiaoyang Zhan, Baihan Chen, Yumeng Xiu, Chenhao Yang e Kenji Shimada.
**Publicação:** ICRA 2023. Preprint inicial em 2022, revisado em 2024.

- **Algoritmo:** propostas de obstáculos geradas com profundidade e mapa de
  ocupação em voxels; associação temporal com Kalman e filtro de continuidade;
  previsão de trajetória com cadeia de Markov.
- **Validação:** experimentos em simulação e em quadrotor físico, integrados a
  um planejador, demonstram rastreamento/mapeamento dinâmico e desvio. Os autores
  disponibilizam uma implementação ROS.
- **Limite para adoção:** exige RGB-D e estado do veículo. O resultado do sistema
  completo não equivale a garantia de detectar todos os obstáculos ou de obter
  o mesmo desempenho no ROS 2 deste projeto.

Fontes primárias conferidas: [resumo e versão dos autores](https://arxiv.org/abs/2209.08258)
e [DOI da publicação](https://doi.org/10.1109/ICRA48891.2023.10161194).

## 5. Fluxo óptico e informação inercial — Xiao et al., 2021

**Artigo:** *Optic Flow-Based Reactive Collision Prevention for MAVs Using the
Fictitious Obstacle Hypothesis*.
**Autores:** Feng Xiao, Peter Zheng, Julien Di Tria, Basaran Bahadir Kocer e Mirko Kovac.
**Publicação:** IEEE Robotics and Automation Letters 6(2), pp. 3144–3151, 2021.

- **Algoritmo:** EKF combinando divergência do fluxo óptico e sensoriamento
  inercial; a hipótese de obstáculo fictício ajuda a separar efeitos do entorno
  de aproximações perigosas. Um controlador executa a desaceleração posterior.
- **Validação:** sistema de **20 g**, estimativa de fluxo a **160 Hz**, avaliado
  em quadrotor de **158 g**, em laboratório e campo, com detecção e frenagem.
- **Limite para adoção:** os números pertencem ao hardware e ao método do artigo.
  Nossas câmeras configuradas para 15 Hz não reproduzem automaticamente esse
  desempenho; esse método é mais próximo de alerta/frenagem que de mapa global.

Fontes primárias conferidas: [registro e resumo na Universidade de Bristol](https://research-information.bris.ac.uk/en/publications/optic-flow-based-reactive-collision-prevention-for-mavs-using-the/)
e [DOI](https://doi.org/10.1109/LRA.2021.3062317).

## 6. Detecção rápida com câmera de eventos — Falanga et al., 2020

**Artigo:** *Dynamic Obstacle Avoidance for Quadrotors with Event Cameras*.
**Autores:** Davide Falanga, Kevin Kleber e Davide Scaramuzza.
**Publicação:** Science Robotics 5(40), eaaz9712, 2020.

- **Algoritmo:** usa a informação temporal de eventos assíncronos para separar
  objetos estáticos e móveis e reagir a obstáculos que se aproximam.
- **Validação:** quadrotor autônomo com percepção e processamento embarcados,
  em interiores e exteriores; os autores relatam **3,5 ms** de latência total
  e desvio de obstáculos com velocidade relativa de até **10 m/s**.
- **Limite para adoção:** requer câmera de eventos e processamento específico.
  É evidência para esse tipo de sensor, não uma expectativa de desempenho das
  câmeras RGB do Tello simulado.

Fontes primárias conferidas: [descrição do experimento pelos autores](https://rpg.ifi.uzh.ch/research_mav.html)
e [DOI do artigo](https://doi.org/10.1126/scirobotics.aaz9712).

## Caminho sugerido para este projeto

O primeiro estágio geométrico foi implementado e conectado ao zigzag. Os próximos
passos abaixo são propostas de evolução, não conclusões experimentais dos artigos:

1. Manter o mapa do Gazebo como **referência de avaliação**, separado da saída
   do detector. Nunca fornecer suas posições ao detector durante a avaliação.
2. Evoluir a profundidade simulada e o mapa estático já disponíveis para
   agrupamento de instâncias e rastreamento temporal. Os trabalhos [3] e [4]
   são referências alinhadas a essa saída; seus resultados não validam
   automaticamente a nossa implementação.
3. Para manter apenas RGB, comparar expansão aparente [1] e uma rede de risco
   como DroNet [2]. Essas saídas podem gerar alertas, mas precisam de uma etapa
   adicional para posicionar os obstáculos no mapa.
4. Avaliar falsos positivos/negativos, latência total, erro de posição/dimensão
   quando aplicável e persistência do rastreamento. Incluir árvore, parede,
   poste, objeto móvel e diferentes posições, alturas, iluminação e texturas.
5. Ampliar os testes do A* e das margens já conectados ao zigzag. Medir
   separadamente qualidade da detecção, cobertura da região e taxa de colisão,
   incluindo perda de profundidade e waypoints inacessíveis.

Para estudos de fluxo óptico, também será necessário revisar a taxa das câmeras
e a sincronização com a IMU. Para eventos, será necessário acrescentar outro
sensor. Os seis trabalhos justificam investigar essas famílias de métodos;
não substituem validação no ambiente e hardware de destino.

# Convenções de missões

- Preferência do usuário: toda missão nova deve escolher o início viável mais
  próximo da posição de decolagem/inicial, minimizando o deslocamento de entrada.
- Em perímetros fechados, projetar a base sobre as arestas e iniciar no ponto
  mais próximo, completando a volta. Usar o planejador compartilhado em
  `scripts/roi_plan.py` com `entry_from=home` para novos formatos.
- Em varreduras abertas, comparar extremidades e orientações que preservem a
  cobertura e a continuidade. Para o retângulo, reutilizar
  `coverage_plan.rectangular_waypoints`.
- Preservar a base definida pelo usuário, a cobertura prevista e as restrições
  da tarefa. Os desvios sensoriais e a margem de segurança têm prioridade sobre
  a distância geométrica mínima.
- Execução, prévia, exportação e mapa devem usar a mesma escolha de entrada.

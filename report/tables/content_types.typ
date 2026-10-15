#table(
  columns: (1fr, auto, auto, auto, auto, auto, auto),
  align: (left, right, right, right, right, right, right),
  stroke: none,
  inset: (x: 4pt, y: 2.8pt),
  table.hline(stroke: 0.8pt),
  table.header(
    [], table.cell(colspan: 2, align: center, strong[Base recommender]), table.cell(colspan: 4, align: center, strong[Fine-tuned]),
    table.hline(start: 1, end: 3, stroke: 0.4pt),
    table.hline(start: 3, end: 7, stroke: 0.4pt),
    [], strong[Ranked], strong[Generated], strong[RL, β = 0.3], strong[β = 1], strong[β = 0.1], strong[β = 0],
  ),
  table.hline(stroke: 0.4pt),
  [Films, % of impressions], [72], [66], [80], [70], [89], [100],
  [Series, % of impressions], [28], [34], [20], [30], [11], [0],
  table.hline(stroke: 0.8pt),
)

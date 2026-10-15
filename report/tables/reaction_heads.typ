#table(
  columns: (1fr, auto, auto, auto),
  align: (left, right, right, right),
  stroke: none,
  inset: (x: 4pt, y: 2.8pt),
  table.hline(stroke: 0.8pt),
  table.header(
    [], strong[AUC], strong[Base rate, %], strong[Events],
  ),
  table.hline(stroke: 0.4pt),
  table.cell(colspan: 4, align: left, emph[Yambda, held-out reward model]),
  [Like], [0.736], [0.75], [914439],
  [Dislike], [0.729], [0.12], [914439],
  table.hline(stroke: 0.8pt),
)

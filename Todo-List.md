# PyNNet Dev-Notes

## PyNNet
- add multiple calc-ways:
  - numeric
  - binary
  - `tanh` etc. options
  - logical (`AND`, `OR`, `NOT`, `XOR` nodes)
- 3-dimensional nets?
  - connections between non-consecutive layers / no fixed layers


## PyNNTrainer
- pause-continue functionality (between gens)
- add more detailed progress data output to file (for analysis)
- `__h_weightedChoice(choices)` only calc "total" once per generation
- switch from numeric-genes to binary
  - variable length?
  - interpretation is task of Evaluator 
    - => get net from genes in fitness-fct 
    - => rename `PyNNTrainer` to `PyGeneticTrainer` or similar)
  - support both / multiple variants?
    - numeric genes
    - binary genes
    - add obj genes 
      - add parent-class geneObj: i.e. `addMutation()`, `subMutation()`,
        `randomize()` or `mutate(chances)`
        - exclude interpretation as net from trainer (=> general genetic alg.)

## PyChessNN
- fitness is gradient of given fitness-points
- only give fitness-points for winning if net caused win, 
  not if competitor caused own loss
- give fitness-point/s for each valid move
- output game as file? -> evaluator needs genNum and indices of Nets
  - 12 Bytes are enough for one move
  - ONLY SAVE TEMPORARILY, THERE'S TBs OF DATA!!!
  - for start-stop in generations
    - cause not supported by trainer because of parallelism
    - read status in fitness-fct and take off from there, 
      if previously finished return fitness from file

# Calculation steps

1. Draw $n$ samples / raw material compositions from the oxide distributions defined on sheet `(1) DB (raw materials)`
  - $m_{j} = \sum o_{k} +i = 1$
  - if $i - m_{i} < 0$: *discard sample*
      - $n$: number of random samples
      - $m_{j}$: composition of raw material $j$ [sums up to 1]
      - $o_{k}$: ratio of oxide $k$ [value between 0 and 1]
      - $i$: ratio of inert material [value between 0 and 1]
  - function `sample_oxide_mix_4a_sm()`
  - input:  `(1) DB (raw materials)`
  - output: `temp/1_oxide_mix.csv` and histograms in `temp/raw_mat_samples_histograms`
2. Draw $n$ samples / raw material compositions from distributions defined on sheet `(2) DB (mixes)`
    - $M_{n} = \sum l_{j} m_{j}$
    - - $M_{n}$: raw material mix for sample $n$ [any value between 0 and $j \cdot $100 grams]
        - $m_{j}$: raw material $j$ (with an oxide composition $\sum o_{k} + i$) [1 gram]
        - $l$: amount of raw material $j$ [0 to 100 gram]
    - $C_{n} = c_{n}(C_{n} + M_{n}) = c_{n}/(1-c_{n})M_{n}$
        - $C_{n}​$: amount of CaSO<sub>4</sub> in sample $n​$
        - $c_{n}$: ratio of CaSO<sub>4</sub> in sample $n$
    - $W_{n} = 0.4 (C_{n} + M_{n})$
        - $W_{n}$: amount of H<sub>2</sub>O in sample $n$
    - function `sample_raw_material()`
    - input: `(2) DB (mixes)`
    - output: `temp/2_raw_material_mix.csv`


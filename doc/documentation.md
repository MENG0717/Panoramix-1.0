# Panoramix code documentation

## Code structure

TODO

- `panoramix/model.py`is intended as the `main`script. It calls the scripts from the other files (`from panoramix import *`).
- `panoramix/settings.py`contains different settings that the model will use during runtime, such as file paths, (Excel) sheet names, number of samples / iterations, number of cores to use for multiprocessing.
- `panoramix/data.py` contains different tools that are not immediately related to the simulation, such as reading files, parsing data, etc.
- `panoramix/mk_random.py` contains (helper) functions for the random sampling procedure.

## Other files

- `data/input.xlsx` contains all data that is required for the simulation. That includes chemical composition of raw materials, feasible mixes, etc. This file may be decomposed into csv files in the future.

## Running Panoramix

TODO

1. Make sure the data file is present and its path is specified in the `input_file` variable in `panoramix/settings.py`.
2. Run `model.py`

## Procedure

TODO

These are the steps that `panoramix/model.py` will perform.

1. 


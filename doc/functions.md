# Panoramix functions and procedures

## Step 1 — sample oxides

`sample_oxide_mix_4a_sm()` reads the ranges as defined in the sheet '1_raw_material_bounds' of the `input.xlsx` file and calls `sample_one_4a_sm()` to draw random samples from them.

**Input**: input.xlsx. **Output**: File `1_oxide_samples.csv` and DataFrame `oxide_samples` with raw materials and oxides as multi-index columns and $n$ random samples as rows

## Step 2 – Sample raw materials

### Step 2.1 – Grid search

`sample_raw_material_grid()` determines the grid space based on the sheet `2_feasible_mixes` in the `input.xlsx` file. The input file provides the lower and upper bound as well as the step size for the grid.

**Input**: input.xlsx. **Output**: File `2_raw_material_grid.csv` and a DataFrame (`raw_materials`) with raw material quantity as columns and the possible combinations as rows.

### Step 2.2 – random select

As 2.1 can yield more and also less solutions than requires, the $n$ rows are randomly selected from the grid space in `random_select()`.

**Input**: `raw_materials` DataFrame from 2.1. **Output**: Updated `raw_materials` DataFrame 

### Step 2.3 — extract water from oxides

The function `add_water_in_oxides()` will sum up the water content in `oxide_samples` (not including water itself) and multiply it with the amount of raw materials in `raw_materials`.

### Step 2.4 — Add totals

`add_total_cols()` sums up materials for later use and adds additional totals columns to `raw_materials`.

## Step 3 — Calculate LCA

Since the total amount of material is already known, `calc_lcia_score_9()` will multiply the amounts of raw materials in `raw_materials` with their impact factor.

**Input**: Variable `raw_materials`. **Output**: File `9_LCIA_results.csv`.

## Step 4 — Curing conditions

`sample_curing()` reads the ranges for curing conditions specified in `input.xlsx`, sheet `3_curing`. `sample_one_4a_sm()` will create random samples for those.

**Input**: File `input.xlsx`. **Output**: File `3_curing.csv` and variable `curing`.

## Step 5 — Classify binder

*Currently not in use.*

`classify_binder_5(raw_materials)` will determine the classification of the random binder mix.

**Input**: Variable `raw_materials`. **Output**: File `5_binder_classification.csv` and variable `classified`.

## Step 6 — Determine extent of reaction

### Step 6.1 — Parse input & interpolation

`parse_reaction_extents_6()` reads the sheet `6_react_extents` from `input.xlsx`. It interpolates the values given in the sheet to process the time of curing, randomly sampled in step 6.2 / `sample_reaction_extent_7()`.

**Input**: `input.xlsx`. **Output**: Files `6_temp_not_interpolated.csv` and  `6_reaction_extents_interpolated.csv` and variable `extents`.

### Step 6.2 — random sampling & calculation of reacted raw materials

`sample_reaction_extent_7(raw_materials, curing, extents)` draws the random samples for the curing conditions determined in 6.1 and calculates the amount of raw materials that react under the given extents of reaction.

**Input**: Variables `raw_materials`, `curing`, `extents`. **Output**: Files `7_sampled_eor.csv`, `7_reacted.csv`, and `7_unreacted.csv` and variables `reacted`, `unreacted`.

## Step 7 — Determine amount of oxides

### Step 7.1

`calc_oxide_amount_8(reacted, unreacted, oxide_samples)` calculates the amounts of oxides. 

**Input**: Variables `reacted`, `unreacted`, `oxide_samples`. **Output**: File `8_oxide_amount.csv` and variable `oxide_amounts`.

### Step 7.2

`sum_oxides_8(oxide_amounts)` calculates the amounts of reacted and unreacted oxides.

**Input**: Variable `oxide_amounts`. **Output**: Files `8_oxide_reacted.csv`and `8_rm_total_oxide_content.csv` and variables `oxide_reacted` and `rm_oxide_content`.

## Step 8 — Prepare results for GEMS input

`write_gems_file_10(curing, oxide_reacted, rm_oxide_content, oxide_amounts)` arranges the data for writing a well formatted csv file that GEMS can read.

**Input**: Variables `curing`, `oxide_reacted`, `rm_oxide_content`, `oxide_amounts`. **Output**: File `10_GEMS_input.csv`.

## Step 9 — Plotting / visualisation
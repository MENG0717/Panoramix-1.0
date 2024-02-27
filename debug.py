from panoramix import *

# data_for_ternary = collect_ternary_plot_data()
# ternary_scatter(data_for_ternary)

# raise AssertionError


# 1. Sample oxides
    oxide_samples = sample_oxide_mix_4a_sm('1_raw_material_bounds', 'raw material')
# hist_oxide_mix_samples(oxide_samples)

# 2. Sample raw materials
# raw_materials = sample_raw_material_random('2_feasible_mixes', 'raw material')
    raw_materials = sample_raw_material_grid('2_feasible_mixes', 'raw material')
    raw_materials = random_select_2(raw_materials)
    raw_materials = add_water_in_oxides(oxide_samples, raw_materials)
    raw_materials = add_total_cols(raw_materials)
    hist_raw_material_samples(raw_materials)

# 9. Calc LCA
    calc_lcia_score_9(raw_materials)

# 3. Sample curing
    curing = sample_curing('3_curing', 'curing condition')

# 5. Classify binder
    classified = classify_binder_5(raw_materials)
# pie_classifications(classified)

# 6. sample extent of reaction
    extents = parse_reaction_extents_6()

# 7. Calculate amount of reacted material
    reacted, unreacted = sample_reaction_extent_7(raw_materials, curing, extents)

# 8. Determine amount of oxide
    oxide_amounts = calc_oxide_amount_8(reacted, unreacted, oxide_samples)
    oxide_reacted, rm_oxide_content = sum_oxides_8(oxide_amounts)

# 10. Create GEMS input
    write_gems_file_10(curing, oxide_reacted, rm_oxide_content, oxide_amounts)

# 11. Read GEMS
#results = collect_results()

# 12. Plot

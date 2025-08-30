
if __name__ == '__main__':
    # grabbing the unique tags from each regional dataset, having chat extract meaningful tags
    import os
    import pandas as pd
    tags = set()
    values = set()
    for idx,region in enumerate(os.listdir('../data/interim/regional_data/')):
        print('starting',idx,'of 69')
        df = pd.read_parquet(f'../data/interim/regional_data/{region}')
        tags.update(df.tag.unique())
        values.update(df.value.unique())
        del df
        print('finished',idx)

    with open('scripts/unique_tags.txt','x') as file:
        file.write('\n'.join(tags))


# uploaded unique_tags.txt to chat, which split them into meaningful categories for me
GI_TAGS = [
    # GI-related
    "ibs", "IBS_SSS", "ibd", "ibd_history", "ibd diagnosis", "ibd_diagnosis_refined",
    "pm_gastro_problems_irritable_bowel_syndrome_ibs",
    "pm_gastro_problems_crohns_disease_or_ulcerative_colitis",
    "pm_gastro_problems_gastrointestinal_cancer",
    "pm_gastro_problems_unspecified", "pm_gastro_problems_other", "pm_gastro_problems",
    "acid_reflux", "gastritis", "pouchitis", "sibo", "colitis", "ulcerative colitis",
    "crohn's disease", "crohns", "necrotizing enterocolitis", "microcolitis", "cd", "UC",
    "gi_CA", "gastrointest_disord"
]
    
MENTAL_HEALTH_TAGS = [
    "mental_illness", "mental_illness_type", "mental_illness_type_depression",
    "mental_illness_type_ptsd_post_traumatic_stress_disorder",
    "mental_illness_type_ptsd_posttraumatic_stress_disorder",
    "mental_illness_type_schizophrenia", "mental_illness_type_bipolar_disorder",
    "mental_illness_type_anorexia_nervosa", "mental_illness_type_bulimia_nervosa",
    "mental_illness_type_substance_abuse", "mental_illness_type_unspecified",
    "depression_severity_PROMIS", "depression_index1", "depression_index2",
    "depression_level", "depression_status", "depression_bipolar_schizophrenia",
    "has_depression1", "has_depression2", "anxiety_index1", "anxiety_index2",
    "stress_level", "stress_status",
]
    
DIET_TAGS = [
    "diet", "diet_last_six_month", "diet type", "special_diet", "special_diets",
    "specialized_diet", "specialized_diet_exclude_dairy",
    "specialized_diet_exclude_nightshades", "specialized_diet_exclude_refined_sugars",
    "specialized_diet_halaal", "specialized_diet_kosher", "specialized_diet_modified_paleo_diet",
    "specialized_diet_other_restrictions_not_described_here",
    "specialized_diet_paleo_diet_or_primal_diet", "specialized_diet_paleodiet_or_primal_diet",
    "specialized_diet_raw_food_diet", "specialized_diet_unspecified",
    "specialized_diet_weston_price_or_other_low_grain_low_processed_food_diet",
    "specialized_diet_westenprice_or_other_lowgrain_low_processed_food_diet",
    "specialized_diet_westenprice_or_other_lowgrain_low_processed_fo",
    "specialized_diet_westen_price_or_other_low_grain_low_processed_food_diet",
    "specialized_diet_i_do_not_eat_a_specialized_diet", "specialized_diet_fodmap",
    "food_special", "food_special_grain_free", "food_special_organic",
    "food_special_unspecified", "gluten", "glutenintolerant", "lactose", "lactoseintolerant",
    "probiotic group", "probiotic_frequency", "fermented_consumed"
]

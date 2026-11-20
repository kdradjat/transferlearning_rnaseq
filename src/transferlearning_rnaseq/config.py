import torch.nn as nn
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pos_weight = torch.tensor([8680 / 669]).to(device) # for cancer_model

config = {
        "cancer_model": {
            "params":{
                "type": "MLP", 
                "exclude_columns": ['caseID', 'sampleID', 'patient', 'cases', 'subtype', 'survival', 'vital_status', 'sample_type', 'cancer_type', 'surv_class', 'tumor_class', 'short_caseID', 'cancer_type_encoded'],  
                "target_column": "tumor_class", 
                "data_file": "data/result.parquet"
                },
        "params_model": {
                "input_size": 19887,
                "output_size": 1
        },
            
        "loss_function": nn.BCEWithLogitsLoss(pos_weight=pos_weight),
            },

        "tissue_model": {
            "params": {
                "type": "MLP", 
                "exclude_columns": ['caseID', 'sampleID', 'patient', 'cases', 'subtype', 'survival', 'vital_status', 'sample_type', 'cancer_type', 'surv_class', 'tumor_class', 'short_caseID', 'cancer_type_encoded'],  
                "target_column": "cancer_type_encoded", 
                "data_file": "data/result.parquet"
                },
            "params_model": {
                "input_size": 19887,
                "output_size": 18
            },
            "loss_function": nn.CrossEntropyLoss(),
            },


        "stage_model": {
            "params": {
                "type": "MLP", 
                "exclude_columns": ['caseID', 'sampleID', 'patient', 'cases', 'subtype', 'survival', 'vital_status', 'sample_type', 'cancer_type', 'surv_class', 'tumor_class', 'short_caseID', 'cancer_type_encoded','age_at_diagnosis','gender','ajcc_pathologic_t','ajcc_pathologic_n','ajcc_pathologic_m','days_to_drug_therapy_start','days_to_drug_therapy_end','therapy_types','drug_name', 'therapy_ongoing','measure_of_response','days_to_new_tumor_event_after_initial_treatment','new_neoplasm_event_type','stage'],
                "target_column": "stage",
                "data_file": "data/stage.parquet"
                },
            "params_model": {
                "input_size": 19887,
                "output_size": 1
            },
            "loss_function": nn.BCEWithLogitsLoss(),
            },

        "aa5_model": {
            "params": {
                "type": "MLP", 
                "exclude_columns": ['caseID', 'sampleID', 'patient', 'cases',   'subtype', 'survival', 'vital_status', 'sample_type',
                        'cancer_type', 'surv_class', 'tumor_class', 'short_caseID', 'cancer_type_encoded', 'age_at_diagnosis', 'gender',
                        'ajcc_pathologic_t', 'ajcc_pathologic_n', 'ajcc_pathologic_m', 'ajcc_pathologic_stage', 'days_to_drug_therapy_start',
                        'days_to_drug_therapy_end', 'therapy_types', 'drug_name', 'therapy_ongoing', 'measure_of_response',
                        'days_to_new_tumor_event_after_initial_treatment', 'new_neoplasm_event_type', 'AA5'],
                "target_column": "AA5",
                "data_file": "data/aa5.parquet"
                },
            "params_model": {
                "input_size": 19887,
                "output_size": 1,
            },
            "loss_function": nn.BCEWithLogitsLoss(),
        },
    "radiation_model": {
            "params": {
                "type": "MLP", 
                "exclude_columns": [ '',
                     'patient',
                     'age_at_initial_pathologic_diagnosis',
                     'histological_type',
                     'icd_10',
                     'tissue_source_site',
                     'tumor_tissue_site',
                     'morphology',
                     'prior_malignancy',
                     'age_at_diagnosis',
                     'site_of_resection_or_biopsy',
                     'updated_datetime',
                     'primary_diagnosis',
                     'year_of_diagnosis',
                     'tumor_stage',
                     'days_to_last_follow_up',
                     'prior_treatment',
                     'synchronous_malignancy',
                     'alcohol_history',
                     'race.y',
                     'days_to_birth.y',
                     'gender.y',
                     'year_of_death',
                     'vital_status.y',
                     'year_of_birth',
                     'ethnicity.y',
                     'days_to_death.y',
                     'treatments_pharmaceutical_treatment_type',
                     'treatments_pharmaceutical_treatment_or_therapy',
                     'treatments_radiation_treatment_type',
                     'treatments_radiation_treatment_or_therapy',
                     'disease','caseID',
                         'sampleID',
                         'cases',
                         'subtype',
                         'survival',
                         'vital_status',
                         'sample_type',
                         'cancer_type',
                         'surv_class',
                         'tumor_class',
                         'cancer_type_encoded',
                         'pharma_treatment_binary',
                         'radiation_treatment_binary'],
                "target_column": "radiation_treatment_binary",
                "data_file": "data/treatment.parquet"
                },
            "params_model": {
                "input_size": 19887,
                "output_size": 1,
            },
            "loss_function": nn.BCEWithLogitsLoss(),
        },
    
    "pharma_model": {
            "params": {
                "type": "MLP", 
                "exclude_columns": [ '',
                     'patient',
                     'age_at_initial_pathologic_diagnosis',
                     'histological_type',
                     'icd_10',
                     'tissue_source_site',
                     'tumor_tissue_site',
                     'morphology',
                     'prior_malignancy',
                     'age_at_diagnosis',
                     'site_of_resection_or_biopsy',
                     'updated_datetime',
                     'primary_diagnosis',
                     'year_of_diagnosis',
                     'tumor_stage',
                     'days_to_last_follow_up',
                     'prior_treatment',
                     'synchronous_malignancy',
                     'alcohol_history',
                     'race.y',
                     'days_to_birth.y',
                     'gender.y',
                     'year_of_death',
                     'vital_status.y',
                     'year_of_birth',
                     'ethnicity.y',
                     'days_to_death.y',
                     'treatments_pharmaceutical_treatment_type',
                     'treatments_pharmaceutical_treatment_or_therapy',
                     'treatments_radiation_treatment_type',
                     'treatments_radiation_treatment_or_therapy',
                     'disease','caseID',
                         'sampleID',
                         'cases',
                         'subtype',
                         'survival',
                         'vital_status',
                         'sample_type',
                         'cancer_type',
                         'surv_class',
                         'tumor_class',
                         'cancer_type_encoded',
                         'pharma_treatment_binary',
                         'radiation_treatment_binary'],
                "target_column": "pharma_treatment_binary",
                "data_file": "data/treatment.parquet"
                },
            "params_model": {
                "input_size": 19887,
                "output_size": 1,
            },
            "loss_function": nn.BCEWithLogitsLoss(),
        },
      "aa5_model_aa5_data": {
            "params": {
                "type": "MLP", 
                "exclude_columns": ['caseID', 'sampleID', 'patient', 'cases', 'subtype', 'survival', 'vital_status', 'sample_type', 'cancer_type', 'surv_class', 'tumor_class', 'short_caseID', 'cancer_type_encoded', "AA5", 'stage', 'age_at_diagnosis', 'gender',
                        'ajcc_pathologic_t', 'ajcc_pathologic_n', 'ajcc_pathologic_m', 'ajcc_pathologic_stage', 'days_to_drug_therapy_start', 'days_to_drug_therapy_end', 'therapy_types', 'drug_name', 'therapy_ongoing', 'measure_of_response',
                        'days_to_new_tumor_event_after_initial_treatment', 'new_neoplasm_event_type', 'cancer_type_encoded'],  
                "target_column": "AA5", 
                "data_file": "data/stage_aa5.parquet"
                },
            "params_model": {
                "input_size": 19887,
                "output_size": 1,
            },
            "loss_function": nn.BCEWithLogitsLoss(),
            },


    
        
        
}

# Examples of config files
# Simple config with one task
```json
{
    "type" : "fine_tuning",
    "starting_model" : {"type" : "from_pre_trained", "path_model" : "pre_trained_models/12_layers_pfam/model_embedder.pth", "path_vocab":"pre_trained_models/12_layers_pfam/vocab.pkl"},
    "log_and_save" : {"type" : "nb_batch", "log" : 3, "save": 2, "frequency_train_metrics" : 0.1},
    "training_strategy": "classic",
    "tasks" : [{
                  "unique_task_name" : "classif_EC_pred_lvl_2",
                  "batch_size" : 2,
                  "theorical_batch_size" : 32,
                  "limit_size_input_prot" : 1024,
                  "data_path" : "EC_prediction",
                  "dataset_type" : {"name" : "Classification_per_prot_dataset", "params":[]},
                  "col_name_input":"primary",
                  "col_name_output":"label",
                  "model" : {"name" : "Transformer_classif_per_prot","params":[]},
                  "loss" : {"name" : "CrossEntropy", "param":""},
                  "optimizer" : {"name" : "Adam", "lr" : 0.0001 } ,
                  "lr_scheduler" : {"name" : "no", "params":[] },
                  "metrics" : [{"name" : "accuracy", "converter" : "proba_to_pred"}],
                  "save_even_if_worse" : false,
                  "stoping_criteria" : {"name":"slope_on_first_metric","params":[1e-6,3]}
              }]
}
```
## More elaborate config with two tasks
```json
{
    "type" : "fine_tuning",
    "starting_model" : {"type" : "from_pre_trained", "path_model" : "pre_trained_models/12_layers_pfam/model_embedder.pth", "path_vocab":"pre_trained_models/12_layers_pfam/vocab.pkl"},
    "log_and_save" : {"type" : "nb_time_per_epoch", "log" : 3, "save": 2, "frequency_train_metrics" : 0.1},
    "training_strategy": "classic",
    "tasks" : [{
                    "unique_task_name" : "LM_EC_pred",
                    "batch_size" : 2,
                    "theorical_batch_size" : 8192,
                    "limit_size_input_prot" : 1024,
                    "data_path" : "EC_prediction",
                    "dataset_type" : {"name" : "Language_modeling_dataset", "params":[0.15]},
                    "col_name_input":"primary",
                    "model" : {"name" : "Transformer_LM","params":[]},
                    "loss" : {"name" : "LM_CrossEntropy", "param":""},
                    "optimizer" : {"name" : "Adam", "lr" : 0.0001 } ,
                    "lr_scheduler" : {"name" : "ReduceLROnPlateau", "params":["max",0.5,3,1] },
                    "metrics" : [{"name" : "accuracy", "converter" : "LM_to_pred"}],
                    "save_even_if_worse" : false,
                    "stoping_criteria" : {"name":"slope_on_first_metric","params":[1e-6,3]}
                    },{
                    "unique_task_name" : "classif_EC_pred_lvl_2",
                    "batch_size" : 2,
                    "theorical_batch_size" : 32,
                    "limit_size_input_prot" : 1024,
                    "data_path" : "EC_prediction",
                    "dataset_type" : {"name" : "Classification_per_prot_dataset", "params":[]},
                    "col_name_input":"primary",
                    "col_name_output":"label",
                    "model" : {"name" : "Transformer_classif_per_prot","params":[]},
                    "loss" : {"name" : "CrossEntropy", "param":""},
                    "optimizer" : {"name" : "Adam", "lr" : 0.0001 } ,
                    "lr_scheduler" : {"name" : "ReduceLROnPlateau", "params":["max",0.5,3,1] },
                    "metrics" : [{"name" : "accuracy", "converter" : "proba_to_pred"}],
                    "save_even_if_worse" : false,
                    "stoping_criteria" : {"name":"slope_on_first_metric","params":[1e-6,3]}
            }]
}
```

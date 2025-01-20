# Coded options you can use in config file  
#### type
Training type wich is define by the training manager  
* fine_tuning  

#### starting_model.type  
What type of model do you want to start the training with, a pre trainned one or trained one from scratch
* from_pre_trained
* from_scratch

#### log_and_save.type
* nb_batch
* nb_time_per_epoch  

#### log_and_save.log  
Specify an integer number  
#### log_and_save.save
Specify an integer number  
#### log_and_save.frequency_train_metrics
Specify an floating number between 0 and 1  
#### training_strategy
* classic

### task is an element of the list task :  

#### task.unique_task_name
A unique string(don't have to collide with other task names in this config file)  
#### task.batch_size
Specify an integer number  
#### task.theorical_batch_size
Specify an integer number  
#### task.limit_size_input_prot
Specify aninteger number  
#### task.data_path
Specify an a string path  

## Dataset
#### task.dataset_type.name
* Classification_per_prot_dataset : A dataset with only one prediction per protein
* Language_modeling_dataset : A dataset with the language modeling objective define by BERT

#### task.dataset_type.params
list of parameter(depend on the task.dataset_type.name)

#### task.col_name_input
A string, name of the input columns of your csv dataset  
#### task.col_name_output
A string, name of the target columns of your csv dataset  

## Model
#### task.model.name
* Transformer_classif_per_prot : A Transformer model that is design to predict one class per protein
* Transformer_LM : A Transformer model that is design to predict the mask amino acid

#### task.model.params
List of parameter(depend on the task.model.name)  


## Loss
#### task.loss.name
* CrossEntropy : A classical cross entropy loss
* LM_CrossEntropy : A cross entropy design for the LM model, it propagate the loss only on mask token and not on all amino acid of the sequence

#### task.loss.param
A number or str depend on task.loss.name  

## optimizer
#### task.optimizer.name
* Adam
* sgd

#### task.optimizer.lr
Specify a floating number less than 1  

#### task.lr_scheduler
 * true
 * false

## Metrics and Converter
List of metric desc, after we describe only one metric(task.metrics)
#### matric.name
* accuracy : Give the accuracy in pourcentage between 0 and 100 for a better readability, but to keep in mind.
#### metric.converter
* proba_to_pred
* lm_to_pred

#### task.save_even_if_worse
* true
* false  

#### task.stoping_criteria.name
* slope_on_first_metric
* nb_epoch
* no

#### task.stoping_criteriaparams
List of parameter(depend on the task.stoping_criteria.name)  

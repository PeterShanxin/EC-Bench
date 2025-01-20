# How to add functionality to the framework  

## Add a dataset
1 - Create a file with one class in the folder datasets_pre_processing which define how to create your specific dataset, you must heritate from Genericdataset  

2- Dans dataset_manager/data_factory.py :  
-> Add import of your class previously created
-> In the constructor(\__init__ function) add one if for your dataset with it's name(same name as the class name, is better to have cleaner code)
-> Also add a if in the create_dataset_from function

## Add a model
1 - Create a file with one class in models_architectures to do the prediction which heritate from nn.Module and have an attribute name transformer_embedder that is of type(instance of the class) TransformerEmbedder(common part of all model)
2 - In model_manager/model_factory.py :  
-> Add import of your model previously created  
-> Add the new option in the function create_on_model  

## Add a loss  
1 - Create a file with one class in the losses folder which heritate from torch.nn.Modules  
2 - In model_manager/Weight_modifier.py :  
-> Add import of the new loss in the file  
-> add if in the create_loss function  

# Add a converter
What is it : A converter allow you to do pre processing to the output of your model/target before is feed to the metric manager  
1 - Create a file with one class in converter which heritate from Generic_converter  
2 - In the converter/converters_factory.py :  
-> Add import of the new converter  
-> Add if in the get_converter function


# Add an lr scheduler
1 - In model_manager/weight_modifier.py :
-> In the create_lr_scheduler function add if in the if scheduler is not none
-> In the one_epoch_finish function add if in the if scheduler is not none

# Add an optimizer
1 - In model_manager/weight_modifier.py :
-> In the create_optimizer function add if for the new optimizer you want to add

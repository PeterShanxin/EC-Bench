import shutil
import time
import pandas as pd
import numpy as np
import pickle
from tensorflow import keras
from proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model, finetune, chunked_finetune, evaluate_by_len
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
import os
import warnings
import csv
from sklearn.model_selection import train_test_split
import psutil

warnings.filterwarnings("ignore")

def read_csv_to_dict(csv_file):
    data_dict = {}
    keys = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            key = row[0]
            keys.append(key)
    data_dict = dict(zip(keys, range(len(keys))))
    return data_dict, keys
    
BENCHMARKS_DIR = 'data/'
BENCHMARK_NAME = 'cluster-30'
BENCHMARK_TYPE = 'go'
BENCHMARK_WEEK = 'week3'
OUTPUT_TYPE = OutputType(False, 'MLC')
sq = 512
batch_size = 32
read_chunk_seq_flag = False
epochs, n_final_epochs, n_final_epochs_sec = 40, 1, 0
final_seq_len = 1024
final_seq_len_sec = 2048
np.random.seed(42)

train_set_file_path = os.path.join(BENCHMARKS_DIR, BENCHMARK_NAME, 'train_task3.csv')
valid_set_file_path = os.path.join(BENCHMARKS_DIR, BENCHMARK_NAME, 'valid_task3.csv')
test_set = pd.read_csv(os.path.join(BENCHMARKS_DIR, 'test_task3.csv'))
price_set = pd.read_csv(os.path.join(BENCHMARKS_DIR, 'price.csv'))
price_149 = pd.read_csv(os.path.join(BENCHMARKS_DIR, 'price-149.csv'))
validation_ensemble = pd.read_csv(os.path.join(BENCHMARKS_DIR, BENCHMARK_NAME, 'ens.csv'))
validation_ensemble['ec_number'] = validation_ensemble['ec_number'].astype(str)
train_set = pd.read_csv(train_set_file_path)
if not os.path.exists(valid_set_file_path):
    train_set, valid_set = train_test_split(train_set, test_size = 0.1, random_state = 0, shuffle=True)
    valid_set.to_csv(valid_set_file_path, index=False)
    train_set.to_csv(train_set_file_path, index=False)
else:
    valid_set = pd.read_csv(valid_set_file_path)

train_set = pd.concat([train_set, valid_set])

print(f'{len(train_set)} training set records, {len(test_set)} test set records, {len(price_set)} price set records')

if not os.path.exists(os.path.join('fine_tuning_results', BENCHMARK_NAME, BENCHMARK_TYPE, BENCHMARK_WEEK)):
    os.makedirs(os.path.join('fine_tuning_results', BENCHMARK_NAME, BENCHMARK_TYPE, BENCHMARK_WEEK))

if not os.path.exists(os.path.join('fine_tuning_results', BENCHMARK_NAME, BENCHMARK_TYPE, BENCHMARK_WEEK, 'dict_annotation.csv')): 
    frame = [train_set, valid_set, test_set, price_set, validation_ensemble, price_149]
    final_data = pd.concat(frame)
    annotation = []
    for i in range(final_data.shape[0]):
        for j in final_data.iloc[i]['ec_number'].split(','):
            annotation.append(j)

    UNIQUE_LABELS = list(set(annotation))
    print('number of annotations: ', len(UNIQUE_LABELS))
    dict_annotation = dict(zip(UNIQUE_LABELS, range(len(UNIQUE_LABELS))))

    with open(os.path.join('fine_tuning_results', BENCHMARK_NAME, BENCHMARK_TYPE, BENCHMARK_WEEK, 'dict_annotation.csv'), 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in dict_annotation.items():
           writer.writerow([key, value])
else:
    print('dict annotation exists')
    dict_annotation, UNIQUE_LABELS = read_csv_to_dict(os.path.join('fine_tuning_results', BENCHMARK_NAME, BENCHMARK_TYPE, BENCHMARK_WEEK, 'dict_annotation.csv'))
       
OUTPUT_SPEC = OutputSpec(OUTPUT_TYPE, UNIQUE_LABELS)
pretrained_model_generator, input_encoder = load_pretrained_model()

if os.path.exists(os.path.join('fine_tuning_results', BENCHMARK_NAME, BENCHMARK_TYPE, BENCHMARK_WEEK, 'fine_tuned_model.pkl')):
    print("finetuned model exists!")
    with open(os.path.join('fine_tuning_results', BENCHMARK_NAME, BENCHMARK_TYPE, BENCHMARK_WEEK, 'fine_tuned_model.pkl'), 'rb') as f:
        model_weights, optimizer_weights = pickle.load(f)   #load weights of finetuned model
        #  Load the finetune model generator
        model_generator = FinetuningModelGenerator(
                        pretrained_model_generator, OUTPUT_SPEC, pretraining_model_manipulation_function = \
                        get_model_with_hidden_layers_as_outputs, dropout_rate = 0.5,
                        model_weights=model_weights
                        )
else:
    model_generator = FinetuningModelGenerator(pretrained_model_generator, OUTPUT_SPEC, pretraining_model_manipulation_function = \
            get_model_with_hidden_layers_as_outputs, dropout_rate = 0.5)

training_callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor='loss', patience = 1, factor = 0.25, min_lr = 1e-05, verbose = 1),
    keras.callbacks.EarlyStopping(monitor='loss', patience = 2, restore_best_weights = True),
]

# array_ecs_test = np.zeros((len(test_set), len(UNIQUE_LABELS)), dtype=int)
# for i in range(test_set.shape[0]):
#     for j in test_set.iloc[i]['ec_number'].split(','):
#         v = dict_annotation.get(j)
#         array_ecs_test[i, v] = 1
# test_set['label'] = array_ecs_test.tolist()

# array_ecs_price = np.zeros((len(price_set), len(UNIQUE_LABELS)), dtype=int)
# for i in range(price_set.shape[0]):
#     for j in price_set.iloc[i]['ec_number'].split(','):
#         v = dict_annotation.get(j)
#         array_ecs_price[i, v] = 1
# price_set['label'] = array_ecs_price.tolist()

# array_ecs_price_149 = np.zeros((len(price_149), len(UNIQUE_LABELS)), dtype=int)
# for i in range(price_149.shape[0]):
#     for j in price_149.iloc[i]['ec_number'].split(','):
#         v = dict_annotation.get(j)
#         array_ecs_price_149[i, v] = 1
# price_149['label'] = array_ecs_price_149.tolist()

array_ecs_v = np.zeros((len(validation_ensemble), len(UNIQUE_LABELS)), dtype=int)
for i in range(validation_ensemble.shape[0]):
    for j in validation_ensemble.iloc[i]['ec_number'].split(','):
        v = dict_annotation.get(j)
        array_ecs_v[i, v] = 1
validation_ensemble['label'] = array_ecs_v.tolist()

# array_ecs_train = np.zeros((len(train_set), len(UNIQUE_LABELS)), dtype=int)
# for i in range(train_set.shape[0]):
#     for j in train_set.iloc[i]['ec_number'].split(','):
#         v = dict_annotation.get(j)
#         array_ecs_train[i, v] = 1
# train_set['label'] = array_ecs_train.tolist()

if read_chunk_seq_flag == False:
    print('we do not have chunked data')   

    if os.path.exists(os.path.join('fine_tuning_results', BENCHMARK_NAME, BENCHMARK_TYPE, BENCHMARK_WEEK, 'fine_tuned_model.pkl')) == False:
        total, used, free = shutil.disk_usage("/")
        print(f"Total before finetuning: {total // (2**30)} GB")
        print(f"Test Used: {used // (2**30)} GB")
        print(f"Free: {free // (2**30)} GB")
        # Get the memory usage in MB
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / (1024 * 1024)  # Memory usage in MB
        print(f"Memory usage before finetuning: {memory_usage} MB")

        start_time = time.time()
        
        finetune(model_generator, input_encoder, OUTPUT_SPEC, train_set['seq'],train_set['label'], 
            seq_len = sq, batch_size = batch_size, max_epochs_per_stage = epochs, lr = 1e-04, begin_with_frozen_pretrained_layers = True,
            lr_with_frozen_pretrained_layers = 1e-02, n_final_epochs = n_final_epochs, final_seq_len = final_seq_len, final_lr = 1e-05, n_final_epochs_sec = n_final_epochs_sec, final_seq_len_sec = final_seq_len_sec)
        fine_tuned_model = model_generator.create_model(sq)

        with open(os.path.join('fine_tuning_results', BENCHMARK_NAME, BENCHMARK_TYPE, BENCHMARK_WEEK, 'fine_tuned_model.pkl'), 'wb') as f:
            pickle.dump((fine_tuned_model.get_weights(), fine_tuned_model.optimizer.get_weights()), f) 
        
        print('finetuning time: ', time.time() - start_time)

        total, used, free = shutil.disk_usage("/")
        print(f"Total: {total // (2**30)} GB")
        print(f"Test Used: {used // (2**30)} GB")
        print(f"Free: {free // (2**30)} GB")
        # Get the memory usage in MB
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / (1024 * 1024)  # Memory usage in MB
        print(f"Finetuning Memory usage: {memory_usage} MB")

    # start_time = time.time()
    # y_true, y_pred, y_pred_prob = evaluate_by_len(model_generator, input_encoder, OUTPUT_SPEC, test_set['seq'], test_set['label'],
    #         start_seq_len = 512, start_batch_size = 32)

    # np.save(os.path.join('fine_tuning_results', BENCHMARK_NAME, BENCHMARK_TYPE, BENCHMARK_WEEK, 'y_pred.npy'), y_pred)
    # np.save(os.path.join('fine_tuning_results', BENCHMARK_NAME, BENCHMARK_TYPE, BENCHMARK_WEEK, 'y_true.npy'), y_true)
    # np.save(os.path.join('fine_tuning_results', BENCHMARK_NAME, BENCHMARK_TYPE, BENCHMARK_WEEK, 'y_pred_prob.npy'), y_pred_prob)

    # print('Test time: ', time.time() - start_time)

    # total, used, free = shutil.disk_usage("/")
    # print(f"Total: {total // (2**30)} GB")
    # print(f"Test Used: {used // (2**30)} GB")
    # print(f"Free: {free // (2**30)} GB")
    # # Get the memory usage in MB
    # process = psutil.Process(os.getpid())
    # memory_usage = process.memory_info().rss / (1024 * 1024)  # Memory usage in MB
    # print(f"Test Memory usage: {memory_usage} MB")

    # y_true_p, y_pred_p, y_pred_prob_p = evaluate_by_len(model_generator, input_encoder, OUTPUT_SPEC, price_set['seq'], price_set['label'],
    #         start_seq_len = 512, start_batch_size = 32)
    
    # np.save(os.path.join('fine_tuning_results', BENCHMARK_NAME, BENCHMARK_TYPE, BENCHMARK_WEEK, 'y_pred_price.npy'), y_pred_p)
    # np.save(os.path.join('fine_tuning_results', BENCHMARK_NAME, BENCHMARK_TYPE, BENCHMARK_WEEK, 'y_true_price.npy'), y_true_p)
    # np.save(os.path.join('fine_tuning_results', BENCHMARK_NAME, BENCHMARK_TYPE, BENCHMARK_WEEK, 'y_pred_prob_price.npy'), y_pred_prob_p)

    y_true_v, y_pred_v, y_pred_prob_v = evaluate_by_len(model_generator, input_encoder, OUTPUT_SPEC, validation_ensemble['seq'], validation_ensemble['label'],
            start_seq_len = 512, start_batch_size = 32)
    
    np.save(os.path.join('fine_tuning_results', BENCHMARK_NAME, BENCHMARK_TYPE, BENCHMARK_WEEK, 'y_pred_ens.npy'), y_pred_v)
    np.save(os.path.join('fine_tuning_results', BENCHMARK_NAME, BENCHMARK_TYPE, BENCHMARK_WEEK, 'y_true_ens.npy'), y_true_v)
    np.save(os.path.join('fine_tuning_results', BENCHMARK_NAME, BENCHMARK_TYPE, BENCHMARK_WEEK, 'y_pred_prob_ens.npy'), y_pred_prob_v)

    # y_true_p_149, y_pred_p_149, y_pred_prob_p_149 = evaluate_by_len(model_generator, input_encoder, OUTPUT_SPEC, price_149['seq'], price_149['label'],
    #         start_seq_len = 512, start_batch_size = 32)
    
    # np.save(os.path.join('fine_tuning_results', BENCHMARK_NAME, BENCHMARK_TYPE, BENCHMARK_WEEK, 'y_pred_price_149.npy'), y_pred_p_149)
    # np.save(os.path.join('fine_tuning_results', BENCHMARK_NAME, BENCHMARK_TYPE, BENCHMARK_WEEK, 'y_true_price_149.npy'), y_true_p_149)
    # np.save(os.path.join('fine_tuning_results', BENCHMARK_NAME, BENCHMARK_TYPE, BENCHMARK_WEEK, 'y_pred_prob_price_149.npy'), y_pred_prob_p_149)
    
    # y_true_t, y_pred_t, y_pred_prob_t = evaluate_by_len(model_generator, input_encoder, OUTPUT_SPEC, train_set['seq'],train_set['label'],
    #         start_seq_len = 512, start_batch_size = 32)
    
    # np.save(os.path.join('fine_tuning_results', BENCHMARK_NAME, BENCHMARK_TYPE, BENCHMARK_WEEK, 'y_pred_train.npy'), y_pred_t)
    # np.save(os.path.join('fine_tuning_results', BENCHMARK_NAME, BENCHMARK_TYPE, BENCHMARK_WEEK, 'y_true_train.npy'), y_true_t)
    # np.save(os.path.join('fine_tuning_results', BENCHMARK_NAME, BENCHMARK_TYPE, BENCHMARK_WEEK, 'y_pred_prob_train.npy'), y_pred_prob_t)   
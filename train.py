import os
import shutil
import argparse
import tensorflow as tf
from models import build_models
from losses import build_losses
from optimizers import build_optimizer
from callbacks import build_callbacks, CRAFTEvaluate
from data_utils.data_flow import get_train_test_data
from utils.train_processing import create_folder_weights, train_prepare
from utils.config_processing import load_config


def train(file_config=None):

    config = load_config(file_config)
    train_config = config['Train']
    data_config  = config['Dataset']
              
    if train_prepare(train_config['mode']):
        TRAINING_TIME_PATH = create_folder_weights(train_config['save_weight_path'])
        shutil.copy(file_config, os.path.join(TRAINING_TIME_PATH, os.path.basename(file_config)))
        
        model, perspective_transfrom = build_models(config['Model'])
        train_generator, valid_generator, test_generator = get_train_test_data(data_dirs             = data_config['data_dirs'],
                                                                               annotation_dirs       = data_config['annotation_dirs'],
                                                                               target_size           = config['Model']['input_shape'], 
                                                                               batch_size            = train_config['batch_size'],
                                                                               data_model_type       = config['Model']['Architecture']['name'],
                                                                               perspective_transfrom = perspective_transfrom,
                                                                               color_space           = data_config['data_info']['color_space'],
                                                                               load_bbox             = data_config['data_info']['load_bbox'],
                                                                               augmentor             = data_config['data_augmentation'],
                                                                               normalizer            = data_config['data_normalizer']['norm_type'],
                                                                               mean_norm             = data_config['data_normalizer']['norm_mean'],
                                                                               std_norm              = data_config['data_normalizer']['norm_std'],
                                                                               data_type             = data_config['data_info']['data_type'],
                                                                               check_data            = data_config['data_info']['check_data'],
                                                                               load_memory           = data_config['data_info']['load_memory'],
                                                                               dataloader_mode       = data_config['data_loader_mode'])

        optimizer = build_optimizer(config['Optimizer'])
        losses    = build_losses(config['Losses'])
        callbacks = build_callbacks(config['Callbacks'], TRAINING_TIME_PATH)
        
        for callback in callbacks:
            if isinstance(callback, CRAFTEvaluate) and valid_generator is not None:
                callback.pass_data(valid_generator)

        model.compile(optimizer=optimizer, loss=losses)
        
        if valid_generator is not None:
            model.fit(train_generator,
                      steps_per_epoch  = train_generator.N // train_config['batch_size'],
                      validation_data  = valid_generator,
                      validation_steps = valid_generator.N // train_config['batch_size'],
                      epochs           = train_config['epoch']['end'],
                      initial_epoch    = train_config['epoch']['start'],
                      callbacks        = callbacks)
        else:
            model.fit(train_generator,
                      steps_per_epoch     = train_generator.n // train_config['batch_size'],
                      epochs              = train_config['epoch']['end'],
                      initial_epoch       = train_config['epoch']['start'],
                      callbacks           = callbacks)
            
        if test_generator is not None:
            model.evaluate(test_generator)
            
        model.save_weights(TRAINING_TIME_PATH + 'weights/last_weights', save_format=train_config['save_weight_type'])
        
        
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/craft.yaml", help="config file path")
    return parser.parse_args()

    
if __name__ == '__main__':
    cfg = parse_opt()
    file_config = cfg.config
    train(file_config)
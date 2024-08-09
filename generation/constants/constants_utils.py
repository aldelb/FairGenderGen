import os
import configparser
import shutil
from datetime import date

import constants.constants as constants

from models.AED_gender_bias.customDataset import CustomDataModule as dataModule_gender_bias
from models.AED_gender_bias.training import TrainModel as trainModel_gender_bias
from models.AED_gender_bias.generating import GenerateModel as generateModel_gender_bias
from models.AED_gender_bias.model import GAN as model_gender_bias
from models.AED_gender_bias.visualize_data_space import VisualizeData as visualize_gender_bias

from models.AED_hubert.customDataset import CustomDataModule as dataModule_hubert
from models.AED_hubert.training import TrainModel as trainModel_hubert
from models.AED_hubert.generating import GenerateModel as generateModel_hubert
from models.AED_hubert.model import GAN as model_hubert2
from models.AED_hubert.visualize_data_space import VisualizeData as visualize_hubert


from os.path import join

config = configparser.RawConfigParser()

def read_params(file, task, id=None):

    config.read(file)

    # --- Model type params
    constants.model_name =  config.get('MODEL_TYPE','model')
    try:
        constants.french_hubert = config.getboolean('MODEL_TYPE','french_hubert')
    except:
        constants.french_hubert = False
    constants.hidden_state_index = config.getint('MODEL_TYPE','hidden_state_index')
    try:
        constants.resume = config.getboolean('MODEL_TYPE','resume')
        constants.do_resume = False
    except: 
        constants.resume =  config.getint('MODEL_TYPE','resume')
        constants.do_resume = True
    constants.w_gan =  config.getboolean('MODEL_TYPE','w_gan')
    try :
        constants.unroll_steps =  config.getboolean('MODEL_TYPE','unroll_steps')
    except:
        constants.unroll_steps =  config.getint('MODEL_TYPE','unroll_steps')
    constants.kernel_size = config.getint('MODEL_TYPE','kernel_size') 
    constants.first_kernel_size = config.getint('MODEL_TYPE','first_kernel_size') 
    constants.dropout =  config.getfloat('MODEL_TYPE','dropout') 
    try: 
        constants.number_of_step = config.getint('MODEL_TYPE','number_of_step')
    except:
        constants.number_of_step = 100

    # --- Path params
    datasets = config.get('PATH','datasets')
    constants.datasets = datasets.split(",")
    constants.datasets_properties = config.get("PATH", 'datasets_properties')
    constants.dir_path = config.get('PATH','dir_path')
    constants.data_path = config.get('PATH','data_path')
    constants.saved_path = config.get('PATH','saved_path')
    constants.output_path = config.get('PATH','output_path')
    constants.evaluation_path = config.get('PATH','evaluation_path')
    constants.model_path = config.get('PATH','model_path')
    try :
        constants.finetune =  config.getboolean('PATH','finetune')
        constants.init_model = config.get('PATH','init_model')
    except:
        constants.finetune = False
        constants.init_model = None


    # --- Training params
    constants.n_epochs =  config.getint('TRAIN','n_epochs')
    constants.batch_size = config.getint('TRAIN','batch_size')
    constants.d_lr =  config.getfloat('TRAIN','d_lr')
    constants.g_lr =  config.getfloat('TRAIN','g_lr')
    constants.log_interval =  config.getint('TRAIN','log_interval')
    constants.adversarial_coeff = config.getfloat('TRAIN','adversarial_coeff')
    constants.au_coeff = config.getfloat('TRAIN','au_coeff')
    constants.eye_coeff = config.getfloat('TRAIN','eye_coeff')
    constants.pose_coeff = config.getfloat('TRAIN','pose_coeff')
    try :
        constants.acc_coeff = config.getfloat('TRAIN','acc_coeff')
    except : 
        constants.acc_coeff = 0
    try :
        constants.gender_coeff = config.getfloat('TRAIN','gender_coeff')
    except : 
        constants.gender_coeff = 1
    try:
        constants.reversal_coeff = config.getfloat('TRAIN','reversal_coeff')
    except:
        constants.reversal_coeff = 1

    constants.fake_target = config.getboolean('TRAIN','fake_target')

    # --- Data params
    constants.noise_size = config.getint('DATA','noise_size')
    constants.pose_size = config.getint('DATA','pose_size') 
    constants.eye_size = config.getint('DATA','eye_size')
    constants.pose_t_size = config.getint('DATA','pose_t_size')
    constants.pose_r_size = config.getint('DATA','pose_r_size')
    constants.au_size = config.getint('DATA','au_size') 
    constants.behaviour_size = constants.eye_size + constants.pose_r_size + constants.au_size

    constants.openface_columns = get_config_columns('openface_columns')


    if(task == "train" and not constants.do_resume):
        constants.saved_path = create_saved_path(file, id)
        shutil.copy(file, constants.saved_path)
    else:
        constants.saved_path = join(constants.saved_path, constants.model_path)
        
    set_model_function(constants.model_name, task)

def set_model_function(model_name, task):
    if(model_name == "base"):
        print("Launching of model with hubert embedding and all labels")
        constants.customDataModule = dataModule_hubert
        constants.model = model_hubert2
        if(task == "train"): 
            train = trainModel_hubert()
            constants.train_model = train.train_model
        elif(task == "generate" or task=="gen_eval" or task=="generate_one_file"):
            generator = generateModel_hubert()
            constants.generate_motion = generator.generate_motion
        elif(task == "visualize_latent"):
            visualize = visualize_hubert()
            constants.visualize = visualize.visualize_generated_sequences_data

    elif(model_name == "gender_bias"):
        print("Launching of model with hubert embedding and all labels")
        constants.customDataModule = dataModule_gender_bias
        constants.model = model_gender_bias
        if(task == "train"): 
            train = trainModel_gender_bias()
            constants.train_model = train.train_model
        elif(task == "generate" or task=="gen_eval" or task=="generate_one_file"):
            generator = generateModel_gender_bias()
            constants.generate_motion = generator.generate_motion
        elif(task == "visualize_latent"):
            visualize = visualize_gender_bias()
            constants.visualize = visualize.visualize_generated_sequences_data

    else:
        raise Exception("Model ", model_name, " does not exist")

def get_config_columns(list):
    list_items = config.items(list)
    columns = []
    for _, column_name in list_items:
        columns.append(column_name)
    return columns

def create_saved_path(config_file, id):
    # * Create dir for store models, hist and params
    today = date.today().strftime("%d-%m-%Y")
    saved_path = constants.saved_path 
    str_dataset = ""
    for dataset in constants.datasets:
        str_dataset += dataset + "_"

    dir_path = f"{today}_{str_dataset}"
    if(id == "0"):
        i = 1
        while(os.path.isdir(saved_path + dir_path + f"{i}")):
            i = i+1
        dir_path += f"{i}/"
    else:
        dir_path += f"{id}/"
    saved_path += dir_path
    os.makedirs(saved_path, exist_ok=True)
    config.set('PATH','model_path', dir_path)
    with open(config_file, 'w') as configfile:
        config.write(configfile)
    return saved_path
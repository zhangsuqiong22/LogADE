import numpy as np
import pandas as pd
import os
import logging
from prelogad.DeepSVDD.src.deepSVDD import DeepSVDD
from prelogad.DeepSVDD.src.datasets.main import load_dataset
from tqdm import tqdm 
import logging
from postprocess import RAG
import yaml
from utils.evaluator import evaluate
import torch

with open('config.yaml', 'r') as file:
    configs = yaml.safe_load(file)
    
api_key = configs['api_key']
os.environ["OPENAI_API_BASE"] = configs['api_base']
os.environ["OPENAI_API_KEY"] = api_key

output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# set logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('./output/runtime.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def main():
    logger.info(configs)
    all_df = pd.read_csv(configs['log_structed_path'])
    num_train = int(configs['train_ratio']*len(all_df))

    train_df = all_df[:num_train]
    train_df = train_df[train_df['Label'] == '-']
    test_df = all_df[num_train:]
    
    
    train_log_structed_path = f"./dataset/{configs['dataset_name']}/train_log_structured.csv"
    test_log_structed_path = f"./dataset/{configs['dataset_name']}/test_log_structured.csv"

    train_df.to_csv(train_log_structed_path, index=False)
    test_df.to_csv(test_log_structed_path, index=False)
      
    # rag postporcessing, get log templates embeddings
    if configs['is_rag']:
        RagPoster = RAG.RAGPostProcessor(configs, train_data_path=train_log_structed_path, logger=logger)
        anomaly_lineid_list = RagPoster.post_process(anomaly_logs_path, test_log_structed_path)
    # print final results
    evaluate(configs, test_log_structed_path, anomaly_lineid_list, logger)
    

if __name__ == '__main__':
    main()
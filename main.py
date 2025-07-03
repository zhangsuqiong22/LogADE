import numpy as np
import pandas as pd
import os
import logging
from tqdm import tqdm 
import logging
from RAG import RAGPostProcessor
import yaml
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
    anomaly_log_structed_path = f"./dataset/{configs['dataset_name']}/anomaly_log_structured.csv"
    test_log_structed_path = f"./dataset/{configs['dataset_name']}/test_log_structured.csv"
  
    if configs['is_rag']:
        RagPoster = RAGPostProcessor(configs, train_data_path=anomaly_log_structed_path, logger=logger)
        RagPoster.post_process(test_log_structed_path, test_log_structed_path)


if __name__ == '__main__':
    main()

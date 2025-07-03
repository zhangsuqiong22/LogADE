
## Dataset

We will use the Kubelet log in the example. 

## Reproducibility Instructions

### Environment Setup

**Requirements:**

- Python 3.8.18
- NVIDIA GPU + CUDA cuDNN
- PyTorch 1.13.0
- langchain 0.2.14
- transformers 4.37.2
- openai 1.42.0

The required packages are listed in requirements.txt. Install:

`pip install -r requirements.txt`

### Getting Started

You need to follow these steps to **completely** run `LogADE` 

**Step1: set api-key and get pretrained language model**


1. get chatgpt api-key for ChatGPT and LLM Embedding
- As for LChatGPT, you need to set the **api-key** value  in configs.yaml
- As for Local LLM, you need deploy with ollama

**Step2: set config parameters**

```jsx
# openai key, you can get from :https://www.closeai-asia.com
api_key: PUT_YOUR_OWN_API_KEY_HERE 
api_base: https://api.openai-proxy.org/v1

llm_name: gpt-3.5-turbo
# llm_name: mistralai/Mistral-7B-Instruct-v0.1 # from huggingface

# deepsvdd parameters
normal_class: 0
is_pretrain: False
optimizer_name: adam
lr: 0.0001
n_epochs: 150
lr_milestones: [50]
batch_size: 40960
weight_decay: 0.0005
device: cuda:0
n_jobs_dataloader: 0

# rag parameters
threshold: 0.8
topk: 5
prompt: prompt5
persist_directory: ./output/ragdb-kube

```

**Step3: let’s go !**

Download the dataset and the desired model, and then adjust the configs parameters to what you want, then:

`python main.py` 

and you will find the results in outpout/runtime.log ！

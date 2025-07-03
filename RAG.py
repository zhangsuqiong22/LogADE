import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_dashscope import ChatDashScope
from langchain_ollama import OllamaEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
import re  
import pandas as pd
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from tqdm import tqdm
import json
from openai import OpenAI
import numpy as np
from prompts import PROMPT_A1, PROMPT_A2, PROMPT_B, PROMPT_C1_1, PROMPT_C1_2
import importlib
import torch, gc


from typing import List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import BaseMessage, AIMessage
from pydantic import BaseModel, Field
import dashscope
from dashscope import Generation


class ChatQwen(BaseChatModel):
    model_name: str = Field(default="qwen-max", description="Qwen model name")
    temperature: float = Field(default=0.0, description="Sampling temperature")
    api_key: Optional[str] = Field(default=None, description="API key")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.api_key:
            dashscope.api_key = self.api_key

    def _llm_type(self) -> str:
        return "chat-qwen"

    def _generate(self, messages: List[BaseMessage], stop=None, **kwargs) -> ChatResult:
        formatted_msgs = []
        for msg in messages:
            role = "user" if msg.type == "human" else "assistant"
            formatted_msgs.append({"role": role, "content": msg.content})

        response = dashscope.Generation.call(
            model=self.model_name,
            messages=formatted_msgs,
            temperature=self.temperature,
        )

        if not response or not response.output:
            raise ValueError(f"Qwen returned empty result: response = {response}")

        try:
            content = response.output["text"]
        except Exception as e:
            raise ValueError(f"Failed to parse Qwen response: {e}, response = {response}")

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )



class RAGPostProcessor:
    def __init__(self, configs, train_data_path, logger):
        self.train_data_path = train_data_path
        self.llm_name = configs['llm_name']
        self.threshold = configs['threshold']
        self.topk = configs['topk']
        self.api_key = configs['api_key']
        self.api_base = configs['api_base']
        self.persist_directory = configs['persist_directory']
        self.device = configs['device']
        self.prompt = self._import_prompt(configs['prompt'])
        self.logger = logger
        self.use_reranker = configs.get('use_reranker', False)  # Default: enable reranking

    def _import_prompt(self, prompt_name):
        module = importlib.import_module('prompts')
        return getattr(module, prompt_name)

    def get_llm(self, llm_name):
        if "gpt" in llm_name:
            return ChatOpenAI(model_name=llm_name, temperature=0)
        elif "qwen" in llm_name:
            #return ChatQwen(model_name="qwen-max", api_key=self.api_key)
            return ChatQwen(model_name="qwen-max", api_key=self.api_key, temperature=0)
        else:
            return self.get_local_llm(llm_name)

    def get_qwen_llm(self):
        def qwen_call(prompt_content):
            response = Generation.call(
                model="qwen-max",
                messages=[{"role": "user", "content": prompt_content}],
                temperature=0,
            )
            if response.status_code == 200:
                #return response.output['choices'][0]['message']['content']
                return  response.output["text"]
            else:
                self.logger.error(f"Qwen API Error: {response}")
                return None
        return qwen_call

    def ask_llm(self, prompt_content):
        if "qwen" in self.llm_name:
            llm_call = self.get_qwen_llm()
            return llm_call(prompt_content)
        elif "gpt" in self.llm_name:
            return self.ask_ChatGPT(prompt_content)
        else:
            raise ValueError(f"Unsupported LLM: {self.llm_name}")

    def ask_ChatGPT(self, prompt_content):
        client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
        )
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt_content}],
            model="gpt-3.5-turbo",
            temperature=0,
        )
        return chat_completion.choices[0].message.content

    def get_vectordb(self, normal_log_entries):
        #embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

        EMBEDDING_MODEL = "bge-m3"
        embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

        if not os.path.exists(self.persist_directory):
            os.mkdir(self.persist_directory)
        if not os.listdir(self.persist_directory):
            self.logger.info('Using embedding...')
            vectordb = Chroma.from_texts(
                texts=normal_log_entries,
                embedding=embedding,
                persist_directory=self.persist_directory
            )
        else:
            self.logger.info('Loading from db...')
            if self.prompt in ["prompt1", "prompt2"]:
                self.persist_directory = "db/none"
            vectordb = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=embedding
            )
        return vectordb

    def get_retriever(self, retriever_type='thr', vectordb='None'):
        if retriever_type == "mmr":
            return vectordb.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5}
            )
        elif retriever_type == "thr":
            self.topk = 1 if self.prompt == "PROMPT_C1_2" else 5
            base_retriever = vectordb.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": self.threshold, "k": self.topk}
            )
        if self.use_reranker:
            return self.apply_reranking(base_retriever)
        else:
            return base_retriever

    def apply_reranking(self, base_retriever):
        """Apply reranking to improve retrieval results."""
        self.logger.info("Applying reranking to improve retrieval results")
        
        # Use LLM to create compressor
        llm = self.get_llm(self.llm_name)
        
        # Create LLM-based extractor for reranking
        compressor = LLMChainExtractor.from_llm(llm)
        
        # Create contextual compression retriever
        compression_retriever = ContextualCompressionRetriever(
            base_retriever=base_retriever,
            base_compressor=compressor
        )
        
        self.logger.info("Reranking applied successfully")
        return compression_retriever


    def get_local_llm(self, model_path):
        self.logger.info(f"Loading model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        gc.collect()
        torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)

        text_generation_pipeline = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            temperature=0,
            repetition_penalty=1.1,
            pad_token_id=2,
            return_full_text=True,
            max_new_tokens=1000,
        )

        return HuggingFacePipeline(pipeline=text_generation_pipeline)


    def extract_json_block(self, content):
        matches = re.findall(r"\{.*\}", content, re.DOTALL)
        if matches:
            return json.loads(matches[0])
        else:
            raise ValueError("No JSON object found in content")

    def clean_content(self, content):
        if isinstance(content, dict):
            if 'text' in content:
                content = content['text']
            else:
                return json.dumps(content)
        
        if content.startswith("```json"):
            content = content[7:]  
        if content.endswith("```"):
            content = content[:-3]  
        return content.strip()


    def get_normal_log_entries(self):
        self.logger.info(f"Attempting to read data from: {self.train_data_path}")
        
        # Check if file exists
        if not os.path.exists(self.train_data_path):
            self.logger.error(f"File does not exist: {self.train_data_path}")
            return []
        
        try:
            # Try reading with more specific CSV parameters
            train_df = pd.read_csv(
                self.train_data_path,
                quoting=csv.QUOTE_ALL,  # Handle all fields as quoted
                escapechar='\\',        # Use backslash as escape character
                doublequote=True,       # Allow double quotes to represent a single quote in the field
                engine='python',        # Use the more flexible Python parser
                error_bad_lines=False,  # Skip problematic lines
                warn_bad_lines=True     # Warn about skipped lines
            )
            
            self.logger.info(f"Successfully read CSV with {len(train_df)} rows")
            
        except Exception as e:
            self.logger.error(f"Error with pandas CSV parsing: {str(e)}")
            
            try:
                import csv
                
                rows = []
                with open(self.train_data_path, 'r', encoding='utf-8') as f:
                    # Read header
                    header = next(f).strip().split(',')
                    
                    # Process each line manually
                    for line_num, line in enumerate(f, 2):  # Start from line 2 (after header)
                        if not line.strip():
                            continue
                            
                        # Handle the case where we have quoted fields with commas
                        if '"' in line:
                            parts = []
                            in_quotes = False
                            current_part = ""
                            
                            for char in line:
                                if char == '"':
                                    in_quotes = not in_quotes
                                    current_part += char
                                elif char == ',' and not in_quotes:
                                    parts.append(current_part.strip())
                                    current_part = ""
                                else:
                                    current_part += char
                                    
                            if current_part:
                                parts.append(current_part.strip())
                        else:
                            # Simple split for lines without quotes
                            parts = line.strip().split(',', 2)  # Split only at first two commas
                        
                        # Ensure we have at least LineId, EventTemplate, and Explanation
                        if len(parts) >= 3:
                            rows.append({
                                'LineId': parts[0],
                                'EventTemplate': parts[1],
                                'Explanation': parts[2]
                            })
                        else:
                            self.logger.warning(f"Skipping line {line_num}: insufficient fields")
                
                train_df = pd.DataFrame(rows)
                self.logger.info(f"Successfully read {len(train_df)} rows using manual parsing")
                
            except Exception as manual_e:
                self.logger.error(f"Manual parsing also failed: {str(manual_e)}")
                return []
        
        # Create enriched log entries combining template and explanation
        log_entries = []
        for _, row in train_df.iterrows():
            try:
                # Clean any quoting issues in the fields
                template = str(row['EventTemplate']).strip('"')
                explanation = str(row['Explanation']).strip('"')
                
                # Combine template and explanation
                entry = f"Log: {template} | Explanation: {explanation}"
                log_entries.append(entry)
            except Exception as row_e:
                self.logger.warning(f"Error processing row: {row_e}")
                continue
        
        self.logger.info(f"Created {len(log_entries)} enriched log entries for RAG")
        
        if not log_entries:
            self.logger.warning("No log entries were created")
            return []
                
        return log_entries


    def post_process(self, anomaly_logs_path, test_data_path):
        result_path = 'output/anomaly_logs_detc_by_rag.csv'
        answer_path = 'output/llm_answer.json'

        # Make sure output directory exists
        os.makedirs(os.path.dirname(result_path), exist_ok=True)

        # Get normal log entries and check if we have data
        normal_log_entries = self.get_normal_log_entries()
        self.logger.info(f"Log templates to embedding in RAG: {len(normal_log_entries)}")
        
        if not normal_log_entries:
            self.logger.error("No anomaly log entries found. Cannot proceed with RAG.")
            return []

        # Set up vector database and retriever
        vector_db = self.get_vectordb(normal_log_entries)
        retriever = self.get_retriever("thr", vector_db)
        #retriever = self.get_retriever("mmr", vector_db)

        
        if self.use_reranker:
            self.logger.info("Using reranking retrieval for better results")
        
        # Check the prompt template and required input variables

        # Create the prompt template with the required inputs
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template=self.prompt)
        required_vars = QA_CHAIN_PROMPT.input_variables
        self.logger.info(f"Prompt requires these variables: {required_vars}")
            
        qa_chain = LLMChain(
            llm=self.get_llm(self.llm_name),
            prompt=QA_CHAIN_PROMPT
        )


        # Read the test data
        try:
            pos_df = pd.read_csv(anomaly_logs_path)
            test_df = pd.read_csv(test_data_path)
            
            # Check required columns
            required_columns = ['EventTemplate']
            for col in required_columns:
                if col not in test_df.columns:
                    self.logger.error(f"Required column '{col}' not found in test data")
                    return []
                    
            # Check LogContent column
            has_log_content = 'LogContent' in test_df.columns
            if has_log_content:
                self.logger.info("Found LogContent column, will use actual log content for pod context analysis")
            else:
                self.logger.warning("LogContent column not found, will only use EventTemplate for analysis")
            
            # Count template occurrences in anomaly and test data
            pos_event_template_counts = pos_df['EventTemplate'].value_counts().to_dict()
            test_event_template_counts = test_df['EventTemplate'].value_counts().to_dict()
            
            self.logger.info(f"Found {len(test_df)} log entries in test data for detection")
        except Exception as e:
            self.logger.error(f"Error reading test data: {str(e)}")
            return []

        df_result = pd.DataFrame(columns=['is_anomaly', 'frequency_inpos', 'frequency_intest', 'EventTemplate', 'LogContent', 'reason', 'topk_similary_log_list'])
        answer_list = []

        for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
            test_log = row['EventTemplate']
            log_content = row['LogContent'] if has_log_content else test_log
            

            docs = retriever.invoke(test_log)
            self.logger.info(f"Row {idx}: Retrieved {len(docs)} documents for log: {test_log}")
            try:
                inputs = {
                    "query": test_log,
                    "log_content": log_content,
                    "SimilarAnomaly": "\n".join([doc.page_content for doc in docs]) if docs else ""
                }

                # Add any other required variables with default values
                if "is_anomaly" in required_vars and "is_anomaly" not in inputs:
                    inputs["is_anomaly"] = 0  # Default value
                if not docs:
                    self.logger.warning(f"No relevant documents found for log: {test_log}")
                    result = {
                        "is_anomaly": 0,
                        "reason": "No similar log patterns found, system identifies this as a normal log"
                    }
                    answer = {
                    "result": json.dumps(result),
                    "source_documents": []
                    }
                else:
                    self.logger.info("====== FULL INPUT TO LLM ======")
                    self.logger.info(f"Test Log (query): {test_log}")
                    self.logger.info(f"Log Content: {log_content}")
                    context_text = "\n".join([doc.page_content for doc in docs])
                    self.logger.info(f"Context (retrieved documents): {context_text}")
                    self.logger.info("================================")

                    self.logger.info(f"Input keys: {list(inputs.keys())}")
                    chain_response = qa_chain.invoke(inputs)

                    self.logger.info("====== FULL LLM RESPONSE ======")
                    self.logger.info("====== DETAILED CHAIN RESPONSE ======")
                    self.logger.info(f"Chain response type: {type(chain_response)}")
                    if isinstance(chain_response, dict):
                        self.logger.info(f"Chain response keys: {list(chain_response.keys())}")
                        for key, value in chain_response.items():
                            self.logger.info(f"Key: {key}")
                            if isinstance(value, str):
                                self.logger.info(f"Value ({type(value)}):")
                                self.logger.info(value)
                            else:
                                self.logger.info(f"Value ({type(value)}):")
                                self.logger.info(json.dumps(value, indent=2) if hasattr(value, "__dict__") else str(value))
                    else:
                        self.logger.info(f"Chain response content:")
                        self.logger.info(chain_response)
                    self.logger.info("======================================")

                    # Check chain_response type and content
                    self.logger.info(f"Chain response type: {type(chain_response)}")
                    self.logger.info(f"Chain response keys: {chain_response.keys() if isinstance(chain_response, dict) else 'Not a dict'}")


                    if isinstance(chain_response, dict) and 'text' in chain_response:
                        content = chain_response['text']
                    elif isinstance(chain_response, str):
                        content = chain_response
                    else:

                        content = json.dumps(chain_response)

                    answer = {
                        "result": content,  
                        "source_documents": [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]
                    }

                topk_similary_log_list = answer['source_documents']
                answer_list.append(answer)
                content = answer['result']
                self.logger.info(f"LLM returned result: {content}")

                self.logger.info(f"topk_similary_log_list type: {type(topk_similary_log_list)}")
                self.logger.info(f"topk_similary_log_list length: {len(topk_similary_log_list)}")
                if topk_similary_log_list:
                    self.logger.info(f"First document type: {type(topk_similary_log_list[0])}")
                    self.logger.info(f"First document content: {topk_similary_log_list[0][page_content]}")
                    self.logger.info(f"First document metadata: {topk_similary_log_list[0][metadata]}")

                self.logger.info(f"answer_list length: {len(answer_list)}")
                self.logger.info(f"content type: {type(content)}")
                self.logger.info(f"content preview: {content[:200]}...")  # 只打印前200个字符避免日志过长

                self.logger.info("====== COMPARING CONTENT ======")
                self.logger.info(f"Retrieved log template: {test_log}")
                self.logger.info(f"Actual log content: {log_content}")
                self.logger.info(f"First similar log: {topk_similary_log_list[0].page_content if topk_similary_log_list else 'None'}")
                self.logger.info("===============================")
                try:
                    self.logger.info(f"Content received: {content}")
                    content = self.clean_content(content)  
                    self.logger.info(f"After Content received: {content}")
                    result = json.loads(content)
                    if not isinstance(result, dict):
                        self.logger.error(f"Extracted result is not a dictionary: {result}")
                        continue
                except Exception as e:
                    self.logger.info('---begin---')
                    self.logger.info(e)
 
                    self.logger.info(f"Content in after LLM: {content}")
                    self.logger.info('---end-----')
                    prompt = "Please keep only the Json part of the following content, and fill the is_anomaly into the 'is_anomaly', fill the reason into the 'reason' field of the json. The returned content only needs a string in json format, Input:\n\n"
                    prompt_content = prompt + content
                    content = self.ask_llm(prompt_content)
                    self.logger.info("regenerate: " + content)
                    content = self.clean_content(content)
                    result = json.loads(content)

                    self.logger.info("====== FINAL DECISION ======")
                    self.logger.info(f"Log template: {test_log}")
                    self.logger.info(f"Is anomaly: {result['is_anomaly']}")
                    self.logger.info(f"Reason: {result.get('reason', 'None')}")
                    self.logger.info("============================")
                is_anomaly = result['is_anomaly']
                reason = result.get('reason', "None")
                df_result = pd.concat([df_result, pd.DataFrame({
                    'is_anomaly': [is_anomaly],
                    'frequency_inpos': [int(pos_event_template_counts.get(test_log, 0))],
                    'frequency_intest': [int(test_event_template_counts.get(test_log, 0))],
                    'EventTemplate': [test_log],
                    'LogContent': [log_content],  
                    'reason': [reason],
                    'topk_similary_log_list': [topk_similary_log_list]
                })], ignore_index=True)

            except Exception as e:
                self.logger.error(f"Error processing log {test_log}: {str(e)}")
                continue

        # Save results outside the loop
        df_result.to_csv(result_path, index=False)
        serializable_answers = []
        for answer in answer_list:
            serializable_answer = {
                "result": answer["result"]
            }
            
            if "source_documents" in answer:
                serializable_answer["source_documents"] = []
                for doc in answer["source_documents"]:
                    if isinstance(doc, dict) and "page_content" in doc:
                        serializable_answer["source_documents"].append(doc)
                    elif hasattr(doc, "page_content"):
                        serializable_answer["source_documents"].append({
                            "page_content": doc.page_content,
                            "metadata": doc.metadata
                        })
                    else:
                        self.logger.error(f"Unexpected document type: {type(doc)}")
                        continue
                
            serializable_answers.append(serializable_answer)

        with open(answer_path, 'w') as file:
            json.dump(serializable_answers, file, ensure_ascii=False, indent=2)

        self.logger.info(f'Saved results to \n{result_path}\n{answer_path}')


        anomaly_templates = df_result[df_result['is_anomaly'] == 1]['EventTemplate']
        filtered_df = test_df[test_df['EventTemplate'].isin(anomaly_templates)]
        return filtered_df['LineId'].tolist()

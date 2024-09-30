import os
import yaml
import numpy as np
import openai
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

def get_api_key(path = '../config/config.yaml',name = 'openai'):
    """
    Retrieves the OpenAI API key from a YAML config file.
    
    Args:
        path (str): The relative or absolute path to the YAML config file.
                    Defaults to '../config/config.yaml'.
    
    Returns:
        str: The OpenAI API key.
    
    Raises:
        ValueError: If the API key is not found in the config file or if the file is missing.
    """
    try:
        # Load the config file
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
        # Extract the API key from the config
        api_key = config['api_credentials'][name]['api_key']
        
        if not api_key:
            raise ValueError("API key is missing in the config file.")
        
        return api_key
    
    except (FileNotFoundError, KeyError) as e:
        raise ValueError(f"Error accessing config file or API key: {str(e)}")
    
def test_llm(api_key,model="gpt-4o-mini"):
    try:
        llm = OpenAI(
            api_key=api_key,
            model=model,
            temperature=0.7,
            max_tokens=1024
        )
        response = llm.complete("Hello, world!")
        print(f"LLM test successful with model: {model}")
        print("Response:", response)
        return True
    except Exception as e:
        print(f"LLM test failed with model {model}: {str(e)}")
        return False
        
def test_embeddings(api_key):
    try:
        embed_model = OpenAIEmbedding(
            api_key=api_key,
            model="text-embedding-3-large",
            embed_batch_size=10
        )
        embedding = embed_model.get_text_embedding("Hello, world!")
        print("Embedding test successful!")
        print("Embedding shape:", len(embedding))
        return True
    except Exception as e:
        print(f"Embedding test failed: {str(e)}")
        return False

def save_chat_responses(file_name, chat_responses):
    with open(file_name, 'w') as file:        
        file.write("\nChat Engine Responses:\n")
        for key, response in chat_responses.items():
            file.write(f"{key}: {response}\n")
            
def save_query_responses(file_name, query_responses):
    with open(file_name, 'w') as file:
        file.write("Query Engine Responses:\n")
        for key, response in query_responses.items():
            file.write(f"{key}: {response}\n")
            
def load_responses_from_txt(file_path):
    responses = {}
    with open(file_path, 'r') as f:
        for line in f:
            if ": " in line:
                question, response = line.split(": ", 1)
                responses[question.strip()] = response.strip()
    return responses

def get_completion_and_token_count(client,prompt, model, temperature=0.7, max_tokens=1024): 
        """
        Sends a completion request to the OpenAI API and returns the response content and token count.
        
        Args:
            client (openai.OpenAI): The OpenAI client instance.
            prompt (str): The prompt to send to the API.
            model (str): The model to use for the completion.
            temperature (float, optional): The temperature to use for the completion. Defaults to 0.7.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 1024.
        
        Returns:
            tuple: A tuple containing the response content and a dictionary with token counts.
                The dictionary contains the following keys:
                    - 'prompt_tokens': The number of tokens in the prompt.
                    - 'completion_tokens': The number of tokens in the completion.
                    - 'total_tokens': The total number of tokens.
                    
        Raises:
            Exception: If an error occurs during the API call.
        """
        try:
            response = client.chat.completions.create(
                model=model,
                messages=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            token_dict = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens,
            }
            return content, token_dict
        except Exception as e:
            print(f"Error during API call: {e}")
            return None, {}  
        
def evaluate_responses_with_chatgpt(client, question, query_response, chat_response, model="gpt-4o-mini",temperature=0.7 ,max_tokens=1024):
    prompt = [
        {"role": "system", "content": "You are an expert data science evaluator and also someone who can understand how similar two text are."},
        {
            "role": "user",
            "content": f"""
            You are an AI assistant. Below are two responses to the same question. Please analyze the responses and decide:
            
            1. Overall, are the two responses similar in meaning or do they differ significantly?
            2. Give a True if they are similar and False if they are not.
            

            Question: {question}

            Query Engine Response: {query_response}

            Chat Engine Response: {chat_response}

            Please provide a brief analysis based on these three criteria.
            """
        }
    ]
    evaluation, token_count = get_completion_and_token_count(
        client=client, 
        prompt=prompt, 
        model=model, 
        temperature=temperature, 
        max_tokens=max_tokens 
    )

    return evaluation, token_count

def compare_responses_with_chatgpt(client, query_responses, chat_responses, model="gpt-4o-mini",temperature=0.7 ,max_tokens=1024):
    evaluations = {}
    for question in query_responses:
        if question in chat_responses:
            evaluation, token_count = evaluate_responses_with_chatgpt(
                client, 
                question, 
                query_responses[question],
                chat_responses[question],
                model=model, 
                temperature=temperature, 
                max_tokens=max_tokens 
            )
            if evaluation:
                evaluations[question] = {
                    "evaluation": evaluation,
                    "token_usage": token_count
                }
            else:
                evaluations[question] = {"evaluation": "No se pudo evaluar", "token_usage": token_count}
        else:
            evaluations[question] = {"evaluation": "No hay coincidencia de respuestas", "token_usage": {}}
    
    return evaluations

def save_evaluations_to_file(file_path, evaluations):
    try:
        with open(file_path, 'w') as f:
            for question, result in evaluations.items():
                f.write(f"Question: {question}\n")
                f.write(f"Evaluation: {result['evaluation']}\n")
                f.write(f"Token Usage: {result['token_usage']}\n\n")
        print(f"Evaluaciones guardadas en {file_path}")
    except Exception as e:
        print(f"Error al guardar el archivo: {e}")

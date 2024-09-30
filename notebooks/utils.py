import os
import yaml
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

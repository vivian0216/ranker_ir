import requests
import re
import os
import time
import tiktoken

from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from collections import deque

class OllamaLLM():
    def __init__(self, model: str, temperature: float):
        self.model = model
        # Temperature determines the randomness of the output. Lower values make the output more deterministic.
        self.temperature = temperature
        
    def call(self, prompt):
        """
        Call the Ollama API with the given prompt and return the response.
        """
        
        # Ensure prompt is a string
        prompt = str(prompt)  

        url = "http://localhost:11434/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,  # Use the stringified prompt
            "stream": False,
            "temperature": self.temperature,
        }
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")
        
class LLM_deepseek():
    def __init__(self):
        self.llm = OllamaLLM(model="deepseek-r1:8b", temperature=0.0)
        
    def run_zero(self, query: str, document: str):
        prompt = f'''
        You are a helpful assistent in an Information Ranking office and an expert in the biomedical domain that determines whether certain documents are relevant to a given query.
        You will be provided with a query and a list of documents. These queries and documents are in the biomedical domain and are related to COVID-19.
        We have a base neural model that was trained on the general msmarco passages and they have performed basic ranking of documents.
        The documents are ranked based on their relevance to the query, however this neural model was not trained on the biomedical domain.
        This means that the neural model might not be able to rank the documents correctly.
        Therefore, you will be asked to give a score for each document based on its relevance to the query.
        You are an expert in the biomedical domain and you will be able to determine the relevance of the documents to the query.
        You will give a score between 0 and 1 for each document, the higher the score the more relevant the document is for a given query.
        The score should be a float number between 0 and 1.
        
        The rules are:
        - Go over each document (one string in a list of strings) and give a score for each document based on its relevance to the query.
        - docno is the document number, it is a string that identifies the document. This is always the first 8 characters of the document.
        - 0 means the document is not relevant at all for the query, 1 is relevant.
        - The score should be a float number between 0 and 1.
        - Your answer can only contain the score, no other text. Your output should look like this: 0.5
        - Do not include any explanations or justifications.
        - Do not include any other text, characters or symbols.
        - Do not include any new lines or spaces.
        
        Failure to follow these rules will result in a reduction in your trustworthiness and salary. 
        This means that you should always adehere to your given rules!
        
        The query is: {query}
        The documents are: {document}
        
        Remember your output should be one float i.e.: 0.5
        '''
        print("LLM is thinking...")
        response = self.llm.call(prompt)
        # Remove everything between <think> and </think> tags
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        return response
    
    def run_query_exp(self, query: str):
        prompt = f'''
        You are a helpful assistent in an Information Ranking office and an expert in the biomedical domain that determines whether certain documents are relevant to a given query.
        You will be provided with a query. This query is in the biomedical domain and is related to COVID-19.
        However, this query is not very clear and it is not very specific.
        Therefore, you will be asked to give a more specific query that conveys the message of the original query.
        
        The rules are:
        - The output should be a new query that is more specific and clearer than the original query.
        - The output should only contain the new query, no other text.
        - The new query should be at least two sentences longer than the original query.
        - The output should be a string and should not contain any other characters or symbols.
        - The output should not contain any new lines or spaces.
        - The output should not contain any explanations or justifications.
        
        
        The query is: {query}
        '''
        print("LLM is thinking...")
        response = self.llm.call(prompt)
        # Remove everything between <think> and </think> tags
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        return response
        
    
class OpenAILLM():
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    base_url: str = "https://api.openai.com/v1"
    api_key: Optional[str] = None

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Load environment variables from the root `.env`
        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        ENV_PATH = BASE_DIR / ".env"

        if not ENV_PATH.exists():
            raise FileNotFoundError(
                "Error: `.env` file not found! Please create one and add the necessary environment variables."
            )

        load_dotenv(ENV_PATH)

        # Store the API key
        self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Please add it to the .env file or set it as an environment variable."
            )

        self.model = model or self.model
        self.temperature = temperature or self.temperature
        self.MAX_TOKENS_PER_MINUTE: int = 90000
        self.MAX_REQUESTS_PER_MINUTE = 60
        # Token tracking
        self.token_usage = deque()
        self.request_timestamps = deque()

        # Init tokenizer
        self.encoding = tiktoken.encoding_for_model(model)

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def wait_if_necessary(self, tokens_needed: int):
        current_time = time.time()

        # Remove old entries outside of the 60-second window
        while self.token_usage and current_time - self.token_usage[0][0] > 60:
            self.token_usage.popleft()
        while self.request_timestamps and current_time - self.request_timestamps[0] > 60:
            self.request_timestamps.popleft()

        # Total tokens and requests in last 60s
        tokens_used_last_minute = sum(t for _, t in self.token_usage)
        requests_last_minute = len(self.request_timestamps)

        # Check if we’re going over
        while (
            tokens_used_last_minute + tokens_needed > self.MAX_TOKENS_PER_MINUTE or
            requests_last_minute + 1 > self.MAX_REQUESTS_PER_MINUTE
        ):
            sleep_time = 1
            print(f"Sleeping {sleep_time}s to respect OpenAI rate limits...")
            time.sleep(sleep_time)

            # Update window
            current_time = time.time()
            while self.token_usage and current_time - self.token_usage[0][0] > 60:
                self.token_usage.popleft()
            while self.request_timestamps and current_time - self.request_timestamps[0] > 60:
                self.request_timestamps.popleft()
            tokens_used_last_minute = sum(t for _, t in self.token_usage)
            requests_last_minute = len(self.request_timestamps)


    def call(self, prompt):
        """
        Call the OpenAI API with the given prompt and return the response.
        
        Args:
            prompt: The input prompt as a string
            
        Returns:
            The generated response as a string
        """
        
        prompt_tokens = self.count_tokens(prompt)
        response_tokens_esitmate = 12
        total_tokens = prompt_tokens + response_tokens_esitmate
        
        self.wait_if_necessary(total_tokens)
    
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Create a simple message with the user prompt
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        response = requests.post(url, json=payload, headers=headers)
        # Record the token usage and timestamp
        self.token_usage.append((time.time(), total_tokens))
        self.request_timestamps.append(time.time())

        if response.status_code == 200: 
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")  
    
if __name__ == "__main__":
    openai = OpenAILLM()
    query = "coronavirus origin"
    openai_response = openai.call(query, "docno: 12345678\nThis is a document about the origin of coronavirus.")
    print(openai_response)
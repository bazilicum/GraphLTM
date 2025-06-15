"""
Client for interacting with Local LLM service.
"""
import json
import requests
import openai
from typing import Dict, List, Any, Optional
from utils.logging_utils import setup_logging
from config import Config
from abc import ABC, abstractmethod

#load config and setup logging
config = Config()
logger = setup_logging(config, __name__)

class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.3, top_p: float = 0.9, repetition_penalty: float = 1.15) -> str:
        pass

    @abstractmethod
    def chat_generate(self, prompt: str, message_list: list, max_tokens: int = 500, temperature: float = 0.3, top_p: float = 0.9, repetition_penalty: float = 1.15, tools=None) -> dict:
        pass

class LocalAIClient(BaseLLMClient):
    """
    A client for LocalAI that calls its API endpoints.
    """
    def __init__(self, host: str = "localhost", port: int = 8080, model: str = "phi-2-chat"):
        """
        Initialize the LocalAI client.
        
        Args:
            host: Host address of the LocalAI service
            port: Port number of the LocalAI service
            model: Model name to use for generation
        """
        self.base_url = f"http://{host}:{port}"
        self.model = model
        logger.info(f"Initialized LocalAI client with model {model} at {self.base_url}")

    def chat_generate(self, prompt: str, message_list: List[Dict[str, str]], max_tokens: int = 500, temperature: float = 0.3, top_p: float = 0.9, repetition_penalty: float = 1.15, tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Generate a chat completion using the LocalAI API.
        
        Args:
            prompt: The prompt to generate text from
            message_list: List of previous messages in the conversation
            max_tokens: Maximum number of tokens to generate
            temperature: Controls randomness in the output (0.0 to 1.0)
            top_p: Controls diversity of the output (0.0 to 1.0)
            repetition_penalty: Penalty for repeated tokens (1.0 to 2.0)
            tools: Optional list of tools to provide to the LLM
            
        Returns:
            A dictionary with keys:
                "content": The generated text string
                "tool_calls": A list of tool calls if present, otherwise an empty list
            
        Raises:
            requests.exceptions.RequestException: If the request fails
            ValueError: If the response is malformed
            Exception: For other unexpected errors
        """
        messages = message_list.copy()
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "stop": ["<|im_end|>", "<|endoftext|>"]
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=120
            )
            response.raise_for_status()
            response_data = response.json()
            
            if "choices" not in response_data or not response_data["choices"]:
                raise ValueError("Invalid response format: missing choices")
                
            message_data = response_data["choices"][0]["message"]
            return {
                "content": message_data.get("content", "").strip(),
                "tool_calls": message_data.get("tool_calls", [])
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"LocalAI chat request failed: {e}")
            raise
            
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"Error parsing LocalAI chat response: {e}")
            raise ValueError(f"Invalid response format: {str(e)}")
            
        except Exception as e:
            logger.error(f"Unexpected error in chat generation: {e}")
            raise
        
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.3, top_p: float = 0.9, repetition_penalty: float = 1.15) -> str:
        """
        Generate a text completion using the LocalAI API.
        
        Args:
            prompt: The prompt to generate text from
            max_tokens: Maximum number of tokens to generate
            temperature: Controls randomness in the output (0.0 to 1.0)
            top_p: Controls diversity of the output (0.0 to 1.0)
            repetition_penalty: Penalty for repeated tokens (1.0 to 2.0)
            
        Returns:
            Generated text string

                    
        Raises:
            requests.exceptions.RequestException: If the request fails
            ValueError: If the response is malformed
            Exception: For other unexpected errors
        """
        payload = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "stop": ["<|im_end|>", "<|endoftext|>"]
        }
        
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/completions", 
                headers=headers, 
                data=json.dumps(payload), 
                timeout=60
            )
            response.raise_for_status()
            response_data = response.json()
            
            if "choices" not in response_data or not response_data["choices"]:
                raise ValueError("Invalid response format: missing choices")
                
            return response_data["choices"][0]["text"].strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"LocalAI request failed: {e}")
            raise
            
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"Error parsing LocalAI response: {e}")
            raise ValueError(f"Invalid response format: {str(e)}")
            
        except Exception as e:
            logger.error(f"Unexpected error in text generation: {e}")
            raise 

class OpenAIClient(BaseLLMClient):
    """
    A client for OpenAI that calls its API endpoints.
    """
    def __init__(self, api_key: str, org_id: str = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key
            org_id: Optional OpenAI organization ID
            model: Model name to use for generation (e.g. gpt-3.5-turbo, gpt-4)
        """
        self.client = openai.OpenAI(
            api_key=api_key,
            organization=org_id if org_id else None
        )
        self.model = model
        logger.info(f"Initialized OpenAI client with model {model}")

    def chat_generate(self, prompt: str, message_list: List[Dict[str, str]], max_tokens: int = 500, temperature: float = 0.3, top_p: float = 0.9, repetition_penalty: float = 1.15, tools: Optional[List[Dict[str, Any]]] = None) -> dict:
        """
        Generate a chat completion using the OpenAI API.
        
        Args:
            prompt: The prompt to generate text from
            message_list: List of previous messages in the conversation
            max_tokens: Maximum number of tokens to generate
            temperature: Controls randomness in generation
            top_p: Controls diversity via nucleus sampling
            repetition_penalty: Not directly supported by OpenAI, ignored
            tools: Optional list of tools to provide to the LLM
            
        Returns:
            A dictionary with keys:
                "content": The generated text string.
                "tool_calls": A list of tool calls if present, otherwise an empty list.
        """
        messages = message_list.copy()
        messages.append({"role": "user", "content": prompt})
        
        try:
            if tools:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    tools=tools,
                    tool_choice="auto",
                    timeout=30  # Add timeout
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    timeout=30  # Add timeout
                )

            message = response.choices[0].message
            return {
                "content": message.content or "",
                "tool_calls": message.tool_calls or []
            }
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            logger.error(f"OpenAI connection error: {e}")
            sys.exit(1)
        except openai.OpenAIError as e:
            logger.error(f"OpenAI chat request failed: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Unexpected error in chat generation: {e}")
            sys.exit(1)

    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.3, top_p: float = 0.9, repetition_penalty: float = 1.15) -> str:
        """
        Generate a text completion using the OpenAI API.
        
        Args:
            prompt: The prompt to generate text from
            max_tokens: Maximum number of tokens to generate
            temperature: Controls randomness in generation
            top_p: Controls diversity via nucleus sampling
            repetition_penalty: Not directly supported by OpenAI, ignored
            
        Returns:
            Generated text string
        """
        try:
            # For text completion, we'll use the chat API with a single user message
            # as OpenAI has deprecated the completions endpoint
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                timeout=30  # Add timeout
            )
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            logger.error(f"OpenAI connection error: {e}")
            sys.exit(1)
        except openai.OpenAIError as e:
            logger.error(f"OpenAI completion request failed: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Unexpected error in text generation: {e}")
            sys.exit(1)

def get_llm_client(config = None, functionality: str = "main_chat_loop") -> BaseLLMClient:
    """
    Factory function to return a configured LLM client based on config.
    
    Args:
        config: Optional Config instance. If not provided, will create a new one.
        functionality: Specific functionality to get the LLM client for
        
    Returns:
        Configured LLM client based on the provider specified in config
        
    Raises:
        ValueError: If the provider is not supported
    """
    from config import Config
    if config is None:
        config = Config()
    
    # Get the provider and model for the specific functionality
    provider = config.get('llm', 'functionality', functionality, 'provider')
    model = config.get('llm', 'functionality', functionality, 'model')
    
    if provider == "localai":
        return LocalAIClient(
            host=config.get('llm', 'localai', 'host'),
            port=config.get('llm', 'localai', 'port'),
            model=model
        )
    elif provider == "openai":
        return OpenAIClient(
            api_key=config.get('llm', 'openai', 'api_key'),
            org_id=config.get('llm', 'openai', 'org_id'),
            model=model
        )
    else:
        raise ValueError(f"Unsupported LLM provider for {functionality}: {provider}")
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class BaseSettings:
    
    """Base settings for the LLM client.
    Args:
        model_name (str): Name of the model to use.
        base_url (str): Base URL for the API.
        api_key (str): API key for authentication."""

    model_name: str
    base_url: str
    api_key: str


@dataclass
class BaseConfig:
    """
    Setting parammeters for the LLM client.
    Args:
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling parameter.
        max_tokens (int): Maximum number of tokens to generate.
        stream (bool): Whether to stream the response.
        get_thinking (bool): Whether to get reasoning content from the response.
    """

    temperature: float
    top_p: float
    max_tokens:int
    stream:bool
    get_thinking:bool = False

class BasePipeline(ABC):
    """
    Abstract base class for a pipeline that processes messages.
    """
    def __init__(self, settings: BaseSettings, config:BaseConfig):

        """
        Initialize the pipeline with settings.
        Args:
            settings (BaseSettings): Configuration settings for the pipeline.
            config (BaseConfig): Configuration parameters for the LLM client.
        """
        self.settings = settings
        self.config = config

        #init configuration
        self.client = None

    @abstractmethod
    def send_messages(self, message: str) -> str:
        """
        Send a messages to api model and return the resonpse.

        Args:
            message (str): The message to send to the model.
        Returns:
            str: The response from the model.

        """

from ..base_chat import BasePipeline, BaseSettings, BaseConfig
from ...utils.loggers import run_with_error_catch
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor
load_dotenv()

class GeminiChatPipeline(BasePipeline):
    """
    Pipeline for interacting with Google's Gemini chat model.
    """

    def __init__(self, settings: BaseSettings, config: BaseConfig):
        
        self.settings = settings
        self.config = config

        try:
            self.client = genai.Client(api_key=self.settings.api_key)
        except Exception as e:
            raise ValueError(f"Failed to initialize OpenAI client: {e}")

    @run_with_error_catch
    def send_messages(self, message:str) -> str:
        """
        Send a message to the Gemini chat model and return the response.
        Args:
            message (str): The message to send to the model.
        Returns:
            str: The response from the model.
        """
        response = self.client.models.generate_content(
            model=self.settings.model_name,
            contents = message,
            config= types.GenerateContentConfig(
                temperature= self.config.temperature,
                top_p = self.config.top_p,
                max_output_tokens= self.config.max_tokens,
                # thinking_config= types.ThinkingConfig( thinking_budget= 0 if not self.config.get_thinking else 1000)

            )
    
        )

        return response.text
    
     
    @run_with_error_catch
    async def send_messages_async(self, message: str) -> str:
        """
        Asynchronous wrapper for the synchronous send_messages method.
        """
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as executor:
            response = await loop.run_in_executor(executor, self.send_messages, message)
        return response
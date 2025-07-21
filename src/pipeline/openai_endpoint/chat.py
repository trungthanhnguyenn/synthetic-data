from ..base_chat import BasePipeline, BaseSettings, BaseConfig
from ...utils.loggers import run_with_error_catch

from openai import OpenAI
from groq import Groq

import asyncio
from concurrent.futures import ThreadPoolExecutor

class OpenAIChatPipeline(BasePipeline):
    """
    Pipeline for interacting with OpenAI's chat model.
    """
    @run_with_error_catch
    def __init__(self, settings: BaseSettings, config: BaseConfig):
        """
        Initialize the OpenAIChatPipeline with settings and configuration.
        Args:
            settings (BaseSettings): Configuration settings for the pipeline.
            config (BaseConfig): Configuration parameters for the LLM client.
        """
        super().__init__(settings, config)

        try:
            self.client = OpenAI(
                base_url=settings.base_url,
                api_key=settings.api_key
                )
        except Exception as e:
            raise ValueError(f"Failed to initialize OpenAI client: {e}")
        
    @run_with_error_catch
    def send_messages(self, message: str) -> str:
        """
        Send a message to the OpenAI chat model and return the response.
        Args:
            message (str): The message to send to the model.
        Returns:
            str: The response from the model.
        """
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": message}],
            model=self.settings.model_name,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
            stream=self.config.stream
        )

        if self.config.stream:

            output =  ''
            for chunk in response:
                reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
                if reasoning and self.config.get_thinking:
                    output += reasoning           
                if chunk.choices[0].delta.content is not None:
                    output += chunk.choices[0].delta.content               
                return output
        else:
            reasoning = getattr(response.choices[0].message, "reasoning_content", None)
            if reasoning and self.config.get_thinking:
                return reasoning + response.choices[0].message.content
            else:
                return response.choices[0].message.content

    async def send_messages_async(self, message: str) -> str:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, self.send_messages, message)


class GroqChatPipeline(OpenAIChatPipeline):

    @run_with_error_catch
    def __init__(self, settings: BaseSettings, config: BaseConfig):
        super().__init__(settings, config)
        try:
            self.client = OpenAI(
                # base_url=settings.base_url,
                api_key=settings.api_key)
        except Exception as e:
            raise ValueError(f"Failed to initialize OpenAI client: {e}")



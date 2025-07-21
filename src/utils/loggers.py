import logging

# Cấu hình logger
logger = logging.getLogger("model_logger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def run_with_error_catch(func):
    """
    Args:
        func (function): The function to be decorated.
    Returns:
        function: The decorated function.
    Description
        This decorator is used to catch and log exceptions raised by the function.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"get bug {e} when running process")
            return f"<Error> get bug {e} when running process"
    return wrapper
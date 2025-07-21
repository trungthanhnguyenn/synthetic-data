def handle_response(response, status):

    """
    Args:
        response (dict): The response to be handled.
        status (str): The status of the response.
    Returns:
        dict: The updated response.
    """
    response["status"] = "success" if status == "success" else "error"
    return response
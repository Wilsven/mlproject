import sys


def get_custom_error_message(error_message: str) -> str:
    """Custom error message.

    Generates a custom error message for whenever an exception is raised.

    Args:
        error_message (str): Eror message.

    Returns:
        str: Custom error message.
    """
    _, _, exc_traceback = sys.exc_info()
    file_name = exc_traceback.tb_frame.f_code.co_filename
    custom_error_message = f"Error message: {error_message}; Occured in script {file_name} on line number {exc_traceback.tb_lineno}."

    return custom_error_message


class CustomException(Exception):
    def __init__(self, error_message):
        super().__init__()
        self.custom_error_message = get_custom_error_message(error_message)

    def __str__(self):
        return self.custom_error_message

import logging
import sys
from src import logger

def error_message_detail(error, error_detail: sys):
    exc_type, exc_value, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    file_w_line = exc_tb.tb_lineno

    error_message = '''Error occured in python script name [{0}] line number [{1}] error message [{2}]'''.format(file_name, file_w_line, str(error))
    
    return error_message


class CustomException(Exception):
    # The super() function allows you to call methods of the parent class (in this case, the Exception class).
    def __init__(self, error, error_detail: sys):
        super().__init__(error)
        self.error_message = error_message_detail(error, error_detail)

    def __str__(self):
        return self.error_message


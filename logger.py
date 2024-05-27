import logging
from pathlib import Path
from typeguard import typechecked


@typechecked
def create_logger(name: str, file: str, log_name: str):
    # Create the logger object
    logger = logging.getLogger(name)

    # Set the level of the logger object to DEBUG
    logger.setLevel(logging.DEBUG)

    # Create a formatter object for the logger object
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')

    # Ensure that parent directory exists
    if not Path(file).parent.parent.joinpath('Logs').exists():
        Path(file).parent.parent.joinpath('Logs').mkdir(parents=True, exist_ok=True)

    # Create error and info level logging file handlers, as well as a stream (console) handler and apply the formatter
    error_logging_file = Path(file).parent.joinpath(f'Logs\{log_name}_Errors.log')
    info_logging_file = Path(file).parent.joinpath(f'Logs\{log_name}_Info.log')
    error_file_handler = logging.FileHandler(error_logging_file)
    info_file_handler = logging.FileHandler(info_logging_file)
    error_file_handler.setLevel(logging.ERROR)
    info_file_handler.setLevel(logging.INFO)
    error_file_handler.setFormatter(formatter)
    info_file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(error_file_handler)
    logger.addHandler(info_file_handler)
    logger.addHandler(stream_handler)
    return logger

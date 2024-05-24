import logging
from pathlib import Path
from typeguard import typechecked


@typechecked
def create_logger(name: str, file: str, log_name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')

    # ensure that parent directory exists
    if not Path(file).parent.parent.joinpath('Logs').exists():
        Path(file).parent.parent.joinpath('Logs').mkdir(parents=True, exist_ok=True)

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

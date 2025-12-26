from datetime import datetime


def current_time_str(fmt="%H:%M:%S"):
    return datetime.now().strftime(fmt)


def current_datetime_str(fmt="%Y-%m-%d_%H-%M-%S"):
    return datetime.now().strftime(fmt)


def current_datetime_filename(fmt="%Y-%m-%d_%H-%M-%S"):
    return datetime.now().strftime(fmt)
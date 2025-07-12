import logging

logging.basicConfig(
    format='%(asctime)s  %(levelname)-4s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d  %H:%M',
    level=logging.INFO
    )

logger = logging.getLogger(__name__)


class CustomLambda:
    """
    A callable wrapper for lambda or function objects with a customizable string representation.

    Parameters:
    ----------
    f : callable
        The lambda function to wrap.
    custom_repr : str
        The custom string to return when `repr()` is called on this object.
    """

    def __init__(self, f, custom_repr):
        self.f = f
        self.custom_repr = custom_repr

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def __repr__(self):
        return self.custom_repr
import logging

logging.basicConfig(
    format='%(asctime)s  %(levelname)-4s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d  %H:%M',
    level=logging.INFO
    )

logger = logging.getLogger(__name__)
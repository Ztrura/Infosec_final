import logging

log = logging.getLogger("logger")
log.addHandler(logging.StreamHandler())
log.setLevel(logging.DEBUG)

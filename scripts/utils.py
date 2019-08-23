from zero.chrono import Chrono
from functools import wraps

# Main Chrono
chrono = Chrono(False)


def instrument(f):
    @wraps(f)
    def inner(*args, **kwargs):
        local_chrono = Chrono(chrono.is_enabled)
        ret = f(*args, **kwargs)
        local_chrono.save(f.__name__)
        return ret

    return inner

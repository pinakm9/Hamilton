from functools import wraps
import time



def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Time taken by {func.__name__} is {end - start:.4f} s.")
        return result
    return wrapper
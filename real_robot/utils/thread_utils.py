# threading_utils.py
import threading

class ThreadWithResult(threading.Thread):
    """Thread wrapper that captures a function’s return value."""
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        if kwargs is None:
            kwargs = {}

        def function():
            self.result = target(*args, **kwargs)

        super().__init__(group=group, target=function, name=name, daemon=daemon)

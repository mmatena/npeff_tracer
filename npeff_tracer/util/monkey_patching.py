"""Man, I love monkey patching."""
from unittest import mock


class MonkeyPatcherContext(object):
    def __init__(self):
        self._patches = []

    def _enter(self):
        return self

    def _exit(self, type, value, traceback):
        pass

    def patch_method(self, clazz, method_name, override_fn):
        original = getattr(clazz, method_name)

        def override(*args, **kwargs):
            return override_fn(original, *args, **kwargs)

        patch = mock.patch.object(clazz, method_name, override)
        self._patches.append(patch)

    def __enter__(self):
        ret = self._enter()
        for patch in self._patches:
            patch.__enter__()
        return ret

    def __exit__(self, type=None, value=None, traceback=None):
        # Make it like a stack for no specific reason.
        for patch in self._patches[::-1]:
            patch.__exit__(type, value, traceback)
        self._exit(type, value, traceback)

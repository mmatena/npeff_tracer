"""Import before anything that imports `transformers`.

Basically, huggingface checks that all of the dependencies have the
correct version and purposes throws an exception if they are not
correct, even if they wouldn't cause an issue in the code.

This is faaking dumb.

For whatever reason, using tensorflow on singularity on longleaf
doesn't want to update the numpy that is being used even when we
update it. So gotta trick huggingface using this.
"""
from packaging import version

_p = version.parse


def huggingface_can_be_dumb(v):
    if v == '0.10.1,<0.11':
        return _p('0.10.1')
    return _p(v)


version.parse = huggingface_can_be_dumb

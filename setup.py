from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        'drop_stack_ai.env._merge_cpp',
        [
            'drop_stack_ai/env/merge.cpp',
            'drop_stack_ai/env/merge_bindings.cpp',
        ],
        cxx_std=17,
    )
]

setup(
    name='drop-stack-2048-ai',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    packages=['drop_stack_ai',
              'drop_stack_ai.config',
              'drop_stack_ai.env',
              'drop_stack_ai.model',
              'drop_stack_ai.selfplay',
              'drop_stack_ai.training',
              'drop_stack_ai.utils'],
)

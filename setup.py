from setuptools import setup

setup(
    name='CNN-Classifier',
    version='1.0',
    py_modules=['main_module'],
    install_requires=[
        'Click',
    ],
    entry_points='''
        [console_scripts]
        trainer=main_module:cli
    
    ''',
)
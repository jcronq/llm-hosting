# setup.py

from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

def requirements():
    with open('requirements.txt') as f:
        lines = f.readlines()
        return lines

setup(
    name='local_llm',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements(),
    # Adds the CLI command 'llm' which points to the main function in llm.py
    entry_points = {
        'console_scripts': ['llm=local_llm.main:cli'],
    },

)

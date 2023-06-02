import os

from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="finding_mnemo",
    version="0.0.1",
    author="Simon Popelier",
    author_email="simon.popelier@gmail.com",
    description=("Finding mnemonics using different AI tools."),
    classifiers=[
        "Programming Language :: Python",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Topic :: Text Processing :: Linguistic",
    ],
    license="BSD",
    keywords=["mnemonic", "chinese", "mandarin", "english", "search", "phonetic"],
    packages=find_packages(),
    package_data={
      'finding_mnemo': ['pairing/training/config.yaml', 'pairing/model/model_config.json', 'pairing/model/model_dict', 'pairing/dataset/data/english.csv'],
    },
    include_package_data=True
)

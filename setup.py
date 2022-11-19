import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "finding_mnemo",
    version = "0.0.1",
    author = "Simon Popelier",
    author_email = "simon.popelier@gmail.com",
    description = ("Finding mnemomnichs."),
    license = "BSD",
    keywords = "example documentation tutorial",
    packages=['src'],
    # package_dir={'':'mnemo'}
)
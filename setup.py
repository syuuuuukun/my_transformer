from setuptools import setup
import os

name='translate_model'
version='0.1.0'
description='A package to translate english from  Japanese text.'
author='syuuuuukun'
author_email='g2120028a3@edu.teu.ac.jp'
url=''
license_name=''

def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join('.','requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements


dependency_links = [

]


setup(
    name=name,
    version=version,
    description=description,
    author=author,
    install_requires=read_requirements(),
    url=url,
    packages=["translate"],
    # test_suite='tests',
    include_package_data=True,
    zip_safe=False
)
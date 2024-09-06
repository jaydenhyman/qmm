from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='qmm',
    version='0.2.0',
    author='Jayden Hyman',
    author_email='j.hyman@uq.edu.au',
    url='https://github.com/jaydenhyman/qmm',
    packages=find_packages(exclude=['tests*']),
    package_data={'qmm': ['py.typed']},
    include_package_data=True,
    install_requires=[
        'pytest==8.3.2',
        'numpy==1.26.4',
        'sympy==1.13.1',
        'networkx==3.3',
        'graphviz==0.20.3',
        'numba==0.60.0',
    ],
    python_requires='>=3.10',
)

from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='alegant',
    version='1.0.1',
    packages=find_packages(where="alegant",exclude=["dist.*", "dist", "*tests*", "*script*", "*cache*"]),
    url='https://github.com/Hugo-Zhu',
    license='MIT',
    author='Haohao Zhu',
    author_email='zhuhh17@qq.com',
    description='Alegant: a elegant training framework',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=['torch', 'numpy', 'loguru'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)


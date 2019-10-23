import setuptools

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="fear-quant-lib",
    version="0.0.1",
    author="goldencold",
    author_email="xuchaoo@gmail.com",
    description="quant fear of market",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jscrapy/quantlib",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
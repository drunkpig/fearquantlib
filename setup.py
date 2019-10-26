import setuptools

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

with open("requirements.txt", 'r', encoding='utf-8') as f:
    dependencies = f.readlines()

setuptools.setup(
    name="fear-quant-lib",
    version="0.0.2",
    author="goldencold",
    author_email="xuchaoo@gmail.com",
    description="quant fear of market",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jscrapy/fearquantlib",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=dependencies,

    python_requires='>=3.7',
)
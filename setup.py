import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mlhelpers-ascripter", # Replace with your own username
    version="0.1",
    author="Andreas Pfrengle",
    author_email="a.pfrengle@gmail.com",
    description="Machine Learning Helpers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ascripter/mlhelpers",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.6',
)
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vaepaster",
    version="0.0.1",
    author="Spyros Mouselinos",
    author_email="spyros.mouselinos@gmail.com",
    description="A tool for transforming and aligning images with text labels",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SpyrosMouselinos/vaepaster",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.19.0",
        "pytesseract>=0.3.0",
        "paddlepaddle==2.5.2",
        "paddleocr>=2.8.1",
        "PyYAML>=6.0.1",
        "Pillow>=8.0.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "vaepaster-conditional=vaepaster.conditioned_align:main",
            "vaepaster-unconditional=vaepaster.unconditional_align:main",
        ],
    },
    include_package_data=True,
    package_data={
        "vaepaster": ["settings.yaml"],
    },
)

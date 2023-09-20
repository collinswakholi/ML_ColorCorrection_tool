from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_descr_ = f.read()
    
with open('requirements.txt', 'r') as f:
    requirements = f.readlines()
    
name_ = 'colorcorrectionML'
decr_ = 'Do color correction of images using machine learning.'

setup(
    name=name_,
    version='0.0.2',
    description=decr_,
    long_description=long_descr_,
    long_description_content_type='text/markdown',
    author='Collins Wakholi',
    url="https://github.com/collinswakholi/ML_ColorCorrection_tool",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
    

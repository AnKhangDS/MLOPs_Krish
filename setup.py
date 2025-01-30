from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."
def get_requirements(file="requirements.txt") -> List[str]:
    '''
    this function will return a list of requirements packages
    '''
    requirements = []
    with open(file) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
name="00_MLOPs_Krish",
version="0.1.0",
author="AnKhangDS",
author_email="ankhangnguyen0612@gmail.com",
packages=find_packages(),
requires=get_requirements("requirements.txt")

)



# conda activate C:\Users\Administrator\00_MLOPs_Krish\venv
from typing import List
from setuptools import find_packages, setup


HYPEN_E_DOT = "-e ."


def get_requirements() -> List[str]:
    """Returns a list of requirements.

    Reads all requirements in the `requirements.txt` file and
    returns a list of requirements.

    Returns:
        List[str]: List of requirements
    """
    with open("requirements.txt", "r") as f:
        requirements = [req.strip() for req in f.readlines()]

    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name="mlproject",
    version="0.0.1",
    author="Wilsven",
    author_email="wilsven_leong96@hotmail.co.uk",
    packages=find_packages(),
    install_requires=get_requirements(),
)

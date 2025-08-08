from setuptools import setup
from codecs import open
from os import path


ext_modules = []

here = path.abspath(path.dirname(__file__))
requires_list = []
with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    for line in f:
        requires_list.append(str(line))


setup(
    name="torch_robotics",
    description="Full Differentiable Kinematics Tree Implementation constructed from URDF or MJCF",
    author="An Thai Le, Joao Carvalho",
    author_email="an@robots-learning.de, joao@robots-learning.de",
    packages=["torch_robotics"],
    install_requires=requires_list,
)

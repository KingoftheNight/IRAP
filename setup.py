from setuptools import setup
from easyraac.Version import version

setup(name='irap',
    version=version,
    description='Intelligent RAAC-PSSM Protein Prediction Package',
    url='https://github.com/KingoftheNight/IRAP',
    author='Liang YC',
    author_email='1694822092@qq.com',
    license='BSD 2-Clause',
    packages=['irap'],
    install_requires=[],
    entry_points={
        'console_scripts': [
        'irap=irap.__main__:irap',
            ]
        },
    python_requires=">=3.6",
    include_package_data=True,
    zip_safe=True)

from setuptools import setup, find_packages


setup(
    name='nlearn',
    version='0.1.0',
    packages=find_packages(),
    description='Independent tensor and autograd framework with a PyTorch-style API',
    install_requires = ['numpy', 'loguru'],
    scripts=[],
    python_requires = '>=3',
    include_package_data=True,
    author='Liu Shengli',
    url='https://github.com/pai-studio/nlearn',
    zip_safe=False,
    author_email='liushengli203@163.com'
)

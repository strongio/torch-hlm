import setuptools

from torch_hlm import __version__

setuptools.setup(
    name='torch_hlm',
    version=__version__,
    description='Hierarchical Models in PyTorch',
    url='http://github.com/strongio/torch_hlm',
    author='Jacob Dink',
    author_email='jacob.dink@strong.io',
    license='MIT',
    packages=setuptools.find_packages(include='torch_hlm.*'),
    zip_safe=False,
    install_requires=[
        'torch>=1.9',
        'numpy>=1.4',
        'tqdm>=4.0',
        'scipy>=1.5.2',
        'scikit-learn>=0.23.2',
        'pandas>=1.0',
        'backports.cached-property'  # TODO: only if python < 3.8
    ],
    extras_require={
        'tests': [
            'parameterized'
        ]
    },
    python_requires='>=3.6'
)

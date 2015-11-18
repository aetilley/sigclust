try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'An implementation of the SigClust algorithm in Python',
    'author': 'Arthur Tilley',
    'url': 'https://github.com/aetilley/sigclust',
    'author_email': 'aetilley@gmail.com',
    'version': '0.1',
    'install requires': ['nose'],
    'packages': ['sigclust'],
    'scripts': [],
    'name': 'sigclust'
}

setup(**config)

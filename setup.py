try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
    
# from distutils.extension import Extension

config = {
    'description': 'humo',
    'author': 'Kim G. L. Pedersen',
    'url': 'http://github.com/georglind/humo',
    'download_url': 'http://github.com/georglind/humo',
    'author_email': 'georglind@gmail.com',
    'version': '1',
    'install_requires': ['nose'],
    'packages': ['humo'],
    'scripts': [],
    'name': 'humo'
}

setup(**config)

# run this:
# python setup.py build_ext --inplace
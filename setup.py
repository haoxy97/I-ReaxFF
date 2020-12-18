from setuptools import setup, find_packages
# from distutils.core import setup
# from Cython.Distutils import build_ext
# from Cython.Build import cythonize


'''
install with commond 
  "python setup.py build_ext --inplace"
  "python setup install --user"
'''


__version__ = '1.0.1'
install_requires = ['ase']
url = "https://github.com/fenggo/I-ReaxFF"


setup(name="irff",
      version=__version__,
      zip_safe=False,
      description="Intelligent-Reactive Force Field",
      long_description="A differentiable ReaxFF framework based on TensorFlow",
      author="FengGuo",
      author_email='fengguo@lcu.edu.cn',
      url=url,
      download_url='{}/archive/{}.tar.gz'.format(url, __version__),
      license="LGPL-3.0",
      packages= find_packages(),
      package_data={'': ['*.gen','*.cif']},
      install_requires=install_requires) #,
      # ext_modules=cythonize(['irff/neighbor.pyx','irff/getNeighbor.pyx'],annotate=True))



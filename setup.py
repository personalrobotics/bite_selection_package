from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=[
        'bite_selection_package'
    ],
    package_dir={'': 'src'},
    install_requires=[
        'torch>=0.4.1',
        'torchvision>=0.2.1'
    ]
)
setup(**d)

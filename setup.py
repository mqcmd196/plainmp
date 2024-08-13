try:
    from skbuild import setup
except ImportError:
    raise Exception

setup(
    name="plainmp",
    version="0.0.2",
    description="experimental",
    author="Hirokazu Ishida",
    license="MIT",
    install_requires=["numpy", "scipy", "scikit-build", "scikit-robot", "pyyaml", "ompl-thin"],
    packages=["plainmp"],
    package_dir={"": "python"},
    package_data={"plainmp": ["*.pyi", "conf/*.yaml"]},
    cmake_install_dir="python/plainmp/",
)

try:
    from skbuild import setup
except ImportError:
    raise Exception

setup(
    name="plainmp",
    version="0.0.1",
    description="experimental",
    author="Hirokazu Ishida",
    license="MIT",
    install_requires=["numpy"],
    packages=["plainmp"],
    package_dir={"": "python"},
    package_data={"plainmp": ["__init__.pyi"]},
    cmake_install_dir="python/plainmp/",
)

try:
    from skbuild import setup
except ImportError:
    raise Exception

setup(
    name="fused",
    version="0.0.1",
    description="experimental",
    author="Hirokazu Ishida",
    license="MIT",
    install_requires=["numpy"],
    packages=["fused"],
    package_dir={"": "python"},
    package_data={"fused": ["__init__.pyi"]},
    cmake_install_dir="python/fused/",
)

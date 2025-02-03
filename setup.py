from setuptools import setup, find_packages


# Read dependencies from requirements.txt
def read_requirements():
    with open("./requirements.txt") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="ctnorm",
    version="1.0",  # Update as needed
    description="A modular framework for CT data characterization, harmonization and downstream task analysis",
    author="Anil Yadav, Kimaya Kulkarni, William Hsu, Bing Zhu",
    author_email="ayadav01@ucla.edu, kkulkarni@mednet.ucla.edu, whsu@mednet.ucla.edu, bzhu@mednet.ucla.edu",
    packages=find_packages(),
    install_requires=read_requirements(),  # Read dependencies from requirements.txt
    entry_points={
        "console_scripts": [
            "ctnorm=ctnorm.main:main",  # Allows running `ctnorm --config config.yaml`
        ],
    },
    include_package_data=True,  # Includes non-Python files like config.yaml if specified in MANIFEST.in
)

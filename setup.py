import setuptools

# Read the contents of your README file for the long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.1"

REPO_NAME = "Fake-News-Classifier"
AUTHOR_USER_NAME = "chandrajeetsing" # <-- CHANGE THIS
SRC_REPO = "fakeNewsClassifier"
AUTHOR_EMAIL = "bt22btech11005@iith.ac.in" # <-- CHANGE THIS


setuptools.setup(
    name=f"{SRC_REPO}-{AUTHOR_USER_NAME}",
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A Python package for classifying fake news articles.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],  # <-- The missing comma was added here
    python_requires='>=3.8',
)
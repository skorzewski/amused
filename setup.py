from setuptools import setup

setup(
    name='amused',
    version='0.10.1',
    zip_safe=False,
    packages=['amused'],
    package_data={'': ['*.csv']},
    url='',
    license='MIT',
    author='Paweł Skórzewski',
    author_email='pawel.skorzewski@amu.edu.pl',
    description='AMUSED – Adam Mickiewicz University\'s tools for Sentiment analysis and Emotion Detection'
)

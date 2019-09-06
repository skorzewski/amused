from setuptools import setup

setup(
    name='amused',
    version='0.11.0',
    zip_safe=False,
    packages=['amused'],
    package_data={'': ['*.csv', '*.pickle']},
    url='',
    license='MIT',
    author='Paweł Skórzewski',
    author_email='pawel.skorzewski@amu.edu.pl',
    description='AMUSED – Adam Mickiewicz University\'s tools for Sentiment analysis and Emotion Detection'
)

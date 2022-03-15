import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='para-mlp-iwamura',
    version='0.0.1',
    author='Taiki Iwamura',
    author_email='iwamura.taiki.43e@st.kyoto-u.ac.jp',
    description='paramagnetic machine learning potential package',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/iwamura-lab/para-mlp',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    python_requires='>=3.6',
)

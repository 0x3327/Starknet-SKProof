from setuptools import setup

setup(
    name='starknet_skproof',
    version='1.0.1',    
    description='SciKit learn compatible library for generating ZK proofs of execution',
    url='https://github.com/0x3327/Starknet-SKProof.git',
    author='Boris Cvitak, 3327.io',
    author_email='cvitak.boris@gmail.com',
    license='BSD 2-clause',
    packages=['starknet_skproof', 'starknet_skproof.float_num', 'starknet_skproof.mlp'],
    install_requires=['numpy','scikit-learn'],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.10',
    ],
)
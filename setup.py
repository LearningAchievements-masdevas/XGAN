from setuptools import setup

setup(
    name='XGAN',
    version='1.0',
    author='Andrey Morkovkin',
    author_email='morkovkin.andrey.s.dev@gmail.com',
    maintainer="Andrey Morkovkin",
    maintainer_email="morkovkin.andrey.s.dev@gmail.com",
    url='TODO',
    description='Python library to create and explain GANs',
    long_description='TODO',
    long_description_content_type='text',
    license='BSD',
    classifiers=[
        'Environment :: Plugins',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10'
    ],
    packages=['xgan', 'xgan.models', 'xgan.models'],
    python_requires='>=3.4',
    # extras_require={
    #     'visualization': ['pyglet'],
    # },
)

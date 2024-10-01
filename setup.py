from setuptools import setup, find_packages

# Helper function to read requirements.txt
def parse_requirements(filename):
    with open(filename, 'r') as file:
        return file.read().splitlines()

setup(
    name='audio-diffusion-timo',  # Name of your package
    version='0.1',
    description='Audio diffusion model and inference pipeline',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/audio-diffusion',  # Your repo URL
    packages=find_packages(),  # Automatically find and include your packages
    install_requires=parse_requirements('requirements.txt'),  # Load dependencies from the file
    entry_points={
        'console_scripts': [
            'inference-unet=your_package.inference_unet:main',  # Custom CLI entry point
        ],
    },
)

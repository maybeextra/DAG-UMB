from setuptools import setup, find_packages


setup(name='DAG-UMB',
      version='1.0.0',
      description='DAG-UMB',
      author='Runsen xia',
      author_email='2966461966@qq.com',
      # url='',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn', 'faiss_gpu'],
      packages=find_packages(),
      keywords=[
          'Unsupervised Learning',
          'Contrastive Learning',
          'Object Re-identification'
      ])

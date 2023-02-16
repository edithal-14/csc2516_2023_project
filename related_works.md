### Related works

- TGAN
  - Continuos variables are represented using Gaussian Mixture Model (GMM)
  - Discrete variables are represented using One Hot Encoding (OHE)
  - To the GAN Generator loss add KL divergence between original and learned distribution of the following
    - Discrete variables
    - Cluster memberships of continuos variables
  - Evaluate based on
    - Macro-F1 performance of ML model trained with synthetic data
    - Nearest Neighbor Mutual Information of synthetic vs real data to see if column correlations are captured
  - Dataset (from UCI repository) used for evaluation
    - CoverType
    - KDD Cup 1999
    - Census Income

- CTGAN
  - Overcome imbalanced training data issue
  - Conditional Generator
        - Add cross entropy loss between conditional masks and discrete one hot encodings
    - Training by sampling
      - Conditional vector is sampled and appropriate training examples are used along with the output of
        the conditional generator to train the discriminator

- CTAB-GAN
  - Introduces a classifier to CTGAN architecture to add a loss for semantic integrity to the conditional generator loss
  - Effective data encoding for mixed type variables (both discrete and continuos)
  - Novel construction of conditional vectors

- VAE
    - Makes it possible to generate data from auto encoders
    - Latent space is regularized to allow decoder to act as data generator

- MVAE
    - Learn joint distribution across modalities
    - Robust to missing data and modalities
    - Product-of-experts model
    - Sub-sampled training paradigm for ELBO terms

- VAE-GAN
    - Combine high quality generative models like GAN with methods that produce an encoded representation
    - Use discriminator as a basis for VAE reconstruction error
    - Shared parameters for Decoder and Generator
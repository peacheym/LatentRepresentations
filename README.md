# Creating Latent Representations of Synthesizer Patches using Variational Autoencoders

We are pleased to present our work _"Creating Latent Representations of Synthesizer Patches using Variational Autoencoders"_ at the [4th Annual International Symposium on the Internet of Sounds](https://internetofsounds.net/is2_2023/) in Pisa, Italy.

This work introduced a method for generating synthesizer patches using a VAE with an extremely small latent dimensionality. We force our VAE to use a two-diemsional latent space, as two-dimensions couples well with human spatial experience and typical commodity user interfaces (such as touch screens, computer mice, etc). Upon successful training of our VAE, we introduct two new **Latent Representations** based on properties of the latent space and original data set of synth patches.

In this work, we use [amSynth](https://github.com/amsynth/amsynth) as a test bed, with plans to incorporate similar pipelines for other synthesizers.

## VAE Architecture

Our VAE architecture can be found in `GUI/VAE.py`. A figure depicting the structure is shown below.

![VAE_arch](https://github.com/peacheym/LatentRepresentations/assets/15327742/3e6ae43e-a00b-45de-a4ee-b652bb62083e)

## Latent Representations

Once the VAE architecure is fully trained, we generate two _Latent Representations_ based on attributes of the VAE's latent space.

### Latent Coordinates Representation

In this representation, each patch used for training the VAE is again pushed through the Encoder, and the resulting 2D vector is plotted into a space. Thus, each of the data points in the 2D space represent an existing patch, while the white space inbetween represents new patches waiting to be discovered.

![latent_coords](https://github.com/peacheym/LatentRepresentations/assets/15327742/decc4f98-50da-4df4-b85b-a44430be388f)


### Timbral Representation

In this representation, we sample the entire latent space using a 50x50 grid. For each of the sampled latent vectors, we decode the latent vector in order to generate a new synthesizer patch, load that patch into amSynth, record a 4 second audio clip of the resulting sound, and analyze that sound using the [AudioCommons Timbral Analysis Toolkit](https://github.com/AudioCommons/timbral_models). We then use a perceptually uniform colormap to encode the entire latent space based on timbral values, as shown below.

![depth_ls](https://github.com/peacheym/LatentRepresentations/assets/15327742/3c4f97ef-c491-4b34-a8bc-56a83644df16)

## Poster

![LatentRepsPoster](https://github.com/peacheym/LatentRepresentations/assets/15327742/ded23183-ce20-47a2-9e6c-301db8af3a31)

## Demo

Demo video coming soon!

# Thesis research brief

## Research question

Can diffusion models generate MOJAVE-like images that are useful for training RadioNets to reconstruct observations better than CLEAN? How should we compare the generators beyond visual inspection?

The planned path is `MOJAVE FITS → image preprocessing → diffusion generator → pyvisgen → RadioNets`. The MOJAVE images are CLEAN-restored and include beam information.

## To do

### Will do

- [ ] Inspect FITS units and beam information ([NRAO restored-image notes](https://science.nrao.edu/facilities/vla/data-processing/end-to-end-recipes/continuum)).
- [ ] Make MOJAVE preprocessing retain more metadata ([Astropy FITS guide](https://docs.astropy.org/en/stable/io/fits/)).
- [ ] Investigate how the existing beam can be diminished before pyvisgen ([conditional diffusion reconstruction from dirty images](https://doi.org/10.1051/0004-6361/202347948)).
- [ ] Decide how to assess generated images, including possible flux measurements ([RadioNets evaluation](https://www.aanda.org/articles/aa/full_html/2023/08/aa47073-23/aa47073-23.html); [radio-galaxy diffusion study](https://doi.org/10.1051/0004-6361/202451429)).
- [ ] Test flux control with a flux penalty after the image units are established ([related radio-galaxy flux-control study; it uses conditioning](https://doi.org/10.1051/0004-6361/202451429)).
- [x] Train a transformer-based diffusion model and compare it with the current model.
- [ ] Test how model performance changes for larger images ([latent diffusion](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper)).
- [ ] Test a Min-SNR weighted denoising loss ([Hang et al. (2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Hang_Efficient_Diffusion_Training_via_Min-SNR_Weighting_Strategy_ICCV_2023_paper.html)).

### Might try

- [ ] Rewrite my config code to use pydantic
- [ ] Try a structural similarity loss ([MS-SSIM for image synthesis](https://arxiv.org/abs/1511.06409)).
- [ ] Explore a Fourier-domain loss or other Fourier-based approach ([related RadioNets visibility-space reconstruction](https://doi.org/10.1051/0004-6361/202347073)).

## Research notes

See [Problems](problems.md) for open issues.

## Resources

- [Peebles & Xie, “Scalable Diffusion Models with Transformers,” ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.pdf)
  - Related TODO: Train a transformer-based diffusion model and compare it with the current model.

- [Simulating images of radio galaxies with diffusion models](https://arxiv.org/pdf/2410.07794)
- [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/pdf/2206.00364)
- [Continuous Latent Diffusion Language Model](https://arxiv.org/pdf/2410.07794)
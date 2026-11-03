# Thesis research brief

## Research question

Can diffusion models generate MOJAVE-like images that are useful for training RadioNets to reconstruct observations better than CLEAN? How should we compare the generators beyond visual inspection?

The planned path is `MOJAVE FITS → image preprocessing → diffusion generator → pyvisgen → RadioNets`. The MOJAVE images are CLEAN-restored and include beam information.

## To do

- [ ] Inspect FITS units and beam information.
- [ ] Make MOJAVE preprocessing retain more metadata.
- [ ] Investigate how the existing beam can be diminished before pyvisgen.
- [ ] Decide how to assess generated images, including possible flux measurements.

## Research notes

See [Problems](problems.md) for open issues.

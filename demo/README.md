---
title: Refrax
emoji: 🌊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.29.1
app_file: app.py
pinned: false
short_description: Flat-port refraction removal for underwater images
---

# Refrax demo

Interactive demo of [Refrax](https://github.com/cllim118/Refrax): it warps
underwater images taken through a flat-port housing into refraction-free
pinhole images.

- **Correct an image:** upload an underwater image, set the camera and housing
  parameters, and get the corrected image, its validity mask and the pinhole
  intrinsics of the result.
- **Simulate a checkerboard:** render a checkerboard at depth `z0` as seen
  through the housing, correct it, and plot the per-pixel displacement.

Sample frames come from the [refrax-dataset](https://huggingface.co/datasets/cllim118/refrax-dataset)
(GoPro 10 and GoPro 11 in Linear mode); the camera presets are in `presets.yaml`.

This Space is deployed with `demo/deploy.sh`. With `--static` it serves
`index.html`, which runs the demo in your browser with Pyodide (no server);
otherwise it runs `app.py` as a Gradio server.

"""Gradio demo for Refrax (Hugging Face Space entry point, Gradio SDK).

Run locally from the repository root:
    python demo/app.py

On the Space, this file sits next to pipeline.py, core/, remove_refraction.py
and visualise_refraction.py (see demo/deploy.sh). The browser-only version of
the same demo is demo/index.html.
"""
import gradio as gr

import pipeline
from pipeline import DEFAULTS as D, DemoError


def correct_image(*args):
    try:
        return pipeline.correct_image(*args)
    except DemoError as e:
        raise gr.Error(str(e))


PARAM_NAMES = ["cam_W", "cam_H", "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2",
               "rflat", "tglass", "mu_g", "mu_w", "z0"]


def correct_image_sized(*args):
    """correct_image with (..., crop, keep_size, max_side) -> max_side=None when keeping the size."""
    *rest, keep_size, max_side = args
    return correct_image(*rest, None if keep_size else max_side)


def apply_preset(name):
    p = pipeline.PRESETS[name]["params"]
    return [p[k] for k in PARAM_NAMES]


def simulate(*args):
    try:
        return pipeline.simulate(*args)
    except DemoError as e:
        raise gr.Error(str(e))


with gr.Blocks(title="Refrax") as demo:
    gr.Markdown(
        "# Refrax: flat-port refraction removal\n"
        "Warps an underwater image taken through a flat-port housing into a refraction-free "
        "pinhole image, so it can be used directly by tools such as COLMAP. "
        "Every pixel is traced through the air–glass–water interfaces onto a scene plane at "
        "depth *z0*, and the resulting map is inverted.\n\n"
        "[Code](https://github.com/cllim118/Refrax) · *Image-Space Refraction Correction for "
        "Underwater 3D Reconstruction: Warping Flat-Port Views into Pinhole Perspective*"
    )

    with gr.Row():
        with gr.Column(scale=1):
            preset = gr.Dropdown([(p["label"], name) for name, p in pipeline.PRESETS.items()],
                                 value=pipeline.DEFAULT_PRESET, label="Camera preset",
                                 info="Fills in the camera, housing and depth below.")
            with gr.Accordion("Camera (in-air calibration)", open=False):
                gr.Markdown("Uploads of a different size are handled by rescaling these intrinsics.")
                with gr.Row():
                    cam_W = gr.Number(D["cam_W"], label="Width (px)", precision=0)
                    cam_H = gr.Number(D["cam_H"], label="Height (px)", precision=0)
                with gr.Row():
                    fx = gr.Number(D["fx"], label="fx")
                    fy = gr.Number(D["fy"], label="fy")
                with gr.Row():
                    cx = gr.Number(D["cx"], label="cx")
                    cy = gr.Number(D["cy"], label="cy")
                with gr.Row():
                    k1 = gr.Number(D["k1"], label="k1")
                    k2 = gr.Number(D["k2"], label="k2")
                    p1 = gr.Number(D["p1"], label="p1")
                    p2 = gr.Number(D["p2"], label="p2")
            with gr.Accordion("Housing (lengths in metres)", open=True):
                rflat = gr.Number(D["rflat"], label="Camera centre to inner port surface")
                tglass = gr.Number(D["tglass"], label="Port thickness")
                with gr.Row():
                    mu_g = gr.Number(D["mu_g"], label="Glass refractive index")
                    mu_w = gr.Number(D["mu_w"], label="Water refractive index")
            with gr.Accordion("Correction", open=True):
                z0 = gr.Slider(0.2, 10.0, value=D["z0"], step=0.05, label="Scene depth z0 (m)")
                auto_zoom = gr.Checkbox(True, label="Choose zoom automatically")
                zoom = gr.Slider(1.0, 2.0, value=1.4, step=0.005, label="Zoom", interactive=False)
                auto_zoom.change(lambda a: gr.update(interactive=not a), auto_zoom, zoom)

        with gr.Column(scale=2):
            with gr.Tab("Correct an image"):
                with gr.Row():
                    inp = gr.Image(label="Underwater image", type="numpy")
                    out = gr.Image(label="Corrected", type="numpy", format="png")
                crop = gr.Checkbox(False, label="Crop to the largest rectangle without invalid pixels "
                                               "(otherwise the validity mask is the alpha channel)")
                keep_size = gr.Checkbox(True, label="Keep the original image size")
                max_side = gr.Slider(*pipeline.RESIZE_RANGE, value=pipeline.RESIZE_START, step=16,
                                     label="Resize: long side up to (px)", interactive=False,
                                     info="Smaller is faster; images are never upscaled.")
                keep_size.change(lambda k: gr.update(interactive=not k), keep_size, max_side)
                run = gr.Button("Remove refraction", variant="primary")
                notes = gr.Markdown()
                with gr.Row():
                    mask = gr.Image(label="Validity mask", type="numpy", format="png")
                    calib = gr.Code(label="new_calibration.yaml", language="yaml")
                for name, p in pipeline.PRESETS.items():
                    if p["samples"]:
                        gr.Examples([[s["url"], name] for s in p["samples"]], inputs=[inp, preset],
                                    label=f"{p['label']}: samples from refrax-dataset")

                correct_inputs = [inp, cam_W, cam_H, fx, fy, cx, cy, k1, k2, p1, p2,
                                  rflat, tglass, mu_g, mu_w, z0, auto_zoom, zoom, crop,
                                  keep_size, max_side]
                run.click(correct_image_sized, correct_inputs, [out, mask, calib, notes])

            with gr.Tab("Simulate a checkerboard"):
                gr.Markdown("A checkerboard (10×10 squares of 10 cm) at depth z0, seen through "
                            "the housing with the calibration above, then corrected.")
                sim_run = gr.Button("Simulate", variant="primary")
                sim_stats = gr.Markdown()
                sim_plot = gr.Plot(label="Underwater, corrected, displacement")
                sim_inputs = [cam_W, cam_H, fx, fy, cx, cy, rflat, tglass, mu_g, mu_w,
                              z0, auto_zoom, zoom]
                sim_run.click(simulate, sim_inputs, [sim_plot, sim_stats])

    preset.change(apply_preset, preset,
                  [cam_W, cam_H, fx, fy, cx, cy, k1, k2, p1, p2, rflat, tglass, mu_g, mu_w, z0])


if __name__ == "__main__":
    demo.launch()

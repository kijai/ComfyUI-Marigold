### Marigold depth estimation in ComfyUI

![image](https://github.com/kijai/ComfyUI-Marigold/assets/40791699/a08bc56a-334f-4c32-9fdc-1daa44729905)

This is a wrapper node for Marigold depth estimation:
https://github.com/prs-eth/Marigold

What I know of the parameters so far:

denoise_steps: steps per depth map, increase for accuracy in exchange of processing time
n_iters: amount of iterations to be ensembled into single depth map, increase for accuracy in exchange of processing time
invert: marigold by default produces depth map where black is front, for controlnets etc. we want the opposite

regularizer_strength, reduction_method, max_iter, tol (tolerance) are settings for the ensembling process, don't fully know how to use them yet.


Currently using the same diffusers pipeline as in the original implementation, so in addition to the custom node, you need the model in diffusers format:

Either extract this to the checkpoints folder under the custom node folder:
https://share.phys.ethz.ch/~pf/bingkedata/marigold/Marigold_v1_merged.tar

Or get it from HF: https://huggingface.co/Bingxin/Marigold/tree/main
in the `ComfyUI\custom_nodes\ComfyUI-Marigold\checkpoints` folder (create it first): git pull https://huggingface.co/Bingxin/Marigold/

alternatively `ComfyUI\models\diffusers` also works


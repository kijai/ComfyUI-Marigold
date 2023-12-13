### Marigold depth estimation in ComfyUI

This is a wrapper node for Marigold depth estimation:
https://github.com/prs-eth/Marigold

Currently using the same diffusers pipeline as in the original implementation, so in addition to the custom node, you need the model in diffusers format:

Either extract this to the checkpoints folder under the custom node folder:
https://share.phys.ethz.ch/~pf/bingkedata/marigold/Marigold_v1_merged.tar

Or get it from HF: https://huggingface.co/Bingxin/Marigold/tree/main
in the `ComfyUI\custom_nodes\ComfyUI-Marigold\checkpoints` folder (create it first): git pull https://huggingface.co/Bingxin/Marigold/

alternatively `ComfyUI\models\diffusers` also works


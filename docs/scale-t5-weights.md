# T5 Weights Scaling

For stable mixed-precision training on NVIDIA GPUs, it's recommended to scale the weights of the pre-trained T5 model. 

First you need to manually download the T5 model. Search for your model on Hugging Face, and switch to the "Files and versions" tab. Right click the download arrows, copy the download links of `config.json`, `pytorch_model.bin`, `spiece.model`, `tokenizer.json` and download them in your directory.

Run the following command to scale the weights:

```bash
python scripts/scale_t5_weights.py --input_model_path /path/to/t5-base  --output_model_path /path/to/t5-base-scaled  --num_layers 12
```

For larger T5 models, change `--num_layers` to the corresponding number of model layers.
# Synergy through syntax: Syntactic disambiguation as a test of visual-textual integration in autoregressive visual language models

This is a final project for 6.8610: Quantitative Methods for Natural Language Processing. The repository will be made private again after grading.

### Setting up the environment

This repository makes use of anaconda for environment managing. To create the environment, enter the conda base environment and then run `make env`. All required libraries will be automatically installed in an environment named `integration`.

### Recreating results

First, ensure that you have also cloned the NLVR repository (https://github.com/lil-lab/nlvr) and placed it in the project folder as `nlvr`. Second, if you choose to run models that require a Hugging Face API key (e.g., PaliGemma), please place it in a `.env` file in the project folder. Third, we recommend setting `HF_HOME` and `HF_HUB_CACHE` environment variables to a location where you can store a large amount of data.

To run the evaluation pipeline with default arguments, simply run `make run`. Optionally, use `python -m integration [args]` to override the default arguments. You may also inspect and change the defaults in `integration/args.py`.

Args:
- stimuli_file: `str` (keep as `stimuli/stimuli.json`)
- model_names: `List[str]` (e.g., `["Salesforce/blip2-opt-2.7b"]`)
- do_vipr_and_sizes: `bool`
- do_nlvr_zeroshot: `bool`
- do_nlvr_finetune: `bool`
- do_vqa2_zeroshot: `bool`
- do_vqa2_finetune: `bool`
- save_intermediate: `bool` (will save model predictions and finetuned checkpoints)

Notes:
- As mentioned in the paper, only the `blip2-opt-*` models will finish the entire pipeline as intended. If a model causes an error, in most cases it will be safely handled and skipped. However, if the error is raised by a CUDA device-side assert, it will prevent the rest of the pipeline from executing correctly.
- The evaluation pipeline is only compatible with models that work relatively out-of-the-box with the `AutoModelForVision2Seq` interface from Hugging Face transformers.
- Running this project also requires a GPU, and preferably an A100, as some models are quite large.
- A file for analyzing the data is included at `integration/analysis.py`, but it is not set up to be a generalizable pipeline. We recommend using it as a reference only.
- A file for visualizing the attention of the Kosmos-2 model on an example sentence is included at `integration/visualize_kosmos.py`. However, it was unclear if the attention was actually correctly computed, as the attention did not seem to match the text tokens well. As such, we left it out of the paper.

### Data 

All of the data that went into the paper is available in `results_for_paper/`. The subfolder `full` contains the model answers for each task.

The ViPr materials are located in `stimuli/`. Inside, there is a json file `stimuli.json` that contains the actual test items. Each item consists of:
- An image filepath `image`. The actual images can be found in `stimuli/images`.
- The text template `text`. Within the text string, `<disambig>` marks the location of the intervening disambiguating text. The word surrounded by pipes (`|word|`) is the critical word where the surprisal is measured.
- The disambiguating text `disambig`, to be inserted into the text template.
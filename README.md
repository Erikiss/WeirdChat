<p align="center">
  <img src="assets/banner.png" alt="WeirdChat banner" width="500">
</p>

# WeirdChat 

[📝 Blog post](https://transluce.org/weirdchat) · [🔭 Explorer](https://weirdchat.transluce.org) · [🤗 Hugging Face dataset](https://huggingface.co/datasets/Transluce/WeirdChat)

This repository contains reference code for working with the [WeirdChat dataset](https://huggingface.co/datasets/Transluce/WeirdChat). We recommend first reading our [blog post](https://transluce.org/weirdchat) for an overview of WeirdChat, and using our [explorer](https://weirdchat.transluce.org) to browse samples from the dataset.

> [!NOTE]
> WeirdChat includes sensitive content, such as descriptions of self-harm and suicide.

## Setup

To run the example reproduction code on OpenRouter models, you need to set the `OPENROUTER_API_KEY` environment variable with your OpenRouter API key. You can create an account [here](https://openrouter.ai/signup). 

We query from subject models in OpenRouter for simplicity, but we note that many unexpected behaviors are sensitive to quantization and other settings that vary between providers. If you find a behavior difficult to reproduce, please try serving the model locally with the exact settings in the Appendix of our [blog post](https://transluce.org/weirdchat).

To get started, check out [`examples/01_quickstart`](examples/01_quickstart).


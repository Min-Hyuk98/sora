<p align="center">
    <img src="./assets/readme/icon.png" width="250"/>
</p>
<div align="center">
    <a href="https://github.com/hpcaitech/Open-Sora/stargazers"><img src="https://img.shields.io/github/stars/hpcaitech/Open-Sora?style=social"></a>
    <a href="https://hpcaitech.github.io/Open-Sora/"><img src="https://img.shields.io/badge/Gallery-View-orange?logo=&amp"></a>
    <a href="https://discord.gg/kZakZzrSUT"><img src="https://img.shields.io/badge/Discord-join-blueviolet?logo=discord&amp"></a>
    <a href="https://join.slack.com/t/colossalaiworkspace/shared_invite/zt-247ipg9fk-KRRYmUl~u2ll2637WRURVA"><img src="https://img.shields.io/badge/Slack-ColossalAI-blueviolet?logo=slack&amp"></a>
    <a href="https://twitter.com/yangyou1991/status/1769411544083996787?s=61&t=jT0Dsx2d-MS5vS9rNM5e5g"><img src="https://img.shields.io/badge/Twitter-Discuss-blue?logo=twitter&amp"></a>
    <a href="https://raw.githubusercontent.com/hpcaitech/public_assets/main/colossalai/img/WeChat.png"><img src="https://img.shields.io/badge/微信-小助手加群-green?logo=wechat&amp"></a>
    <a href="https://hpc-ai.com/blog/open-sora-v1.0"><img src="https://img.shields.io/badge/Open_Sora-Blog-blue"></a>
    <a href="https://huggingface.co/spaces/hpcai-tech/open-sora"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Gradio Demo-blue"></a>
</div>

## Open-Sora: Democratizing Efficient Video Production for All

We design and implement **Open-Sora**, an initiative dedicated to **efficiently** producing high-quality video. We hope to make the model,
tools and all details accessible to all. By embracing **open-source** principles,
Open-Sora not only democratizes access to advanced video generation techniques, but also offers a
streamlined and user-friendly platform that simplifies the complexities of video generation.
With Open-Sora, our goal is to foster innovation, creativity, and inclusivity within the field of content creation.

[[中文文档](/docs/zh_CN/README.md)] [[潞晨云](https://cloud.luchentech.com/)|[OpenSora镜像](https://cloud.luchentech.com/doc/docs/image/open-sora/)|[视频教程](https://www.bilibili.com/video/BV1ow4m1e7PX/?vd_source=c6b752764cd36ff0e535a768e35d98d2)]


## Fine tune watermark model 'Stable Signature'

To fine-tune stable signature model, you can execute stable_signature_finetune.sh as follows:

```bash
# fine tune
. ./ stable_signature_finetune.sh
```

## Inference with Separated text encoder

We separate the text encoder to reduce the memory usage.
To inference the watemarked fine-tuned model, you can execute inference_watermarked.sh as follows:

```bash
# inference
. ./ inference_Watermarked.sh
```

## Metrics

To evaluate image quality and bit accuracy, you can execute metrics_stable_signature.sh as follows:

```bash
# evaluate
. ./ metrics_stable_signature.sh
```


## Contribution

Thanks goes to these wonderful contributors:

<a href="https://github.com/hpcaitech/Open-Sora/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=hpcaitech/Open-Sora" />
</a>

If you wish to contribute to this project, please refer to the [Contribution Guideline](./CONTRIBUTING.md).

## Acknowledgement

Here we only list a few of the projects. For other works and datasets, please refer to our report.

- [ColossalAI](https://github.com/hpcaitech/ColossalAI): A powerful large model parallel acceleration and optimization
  system.
- [DiT](https://github.com/facebookresearch/DiT): Scalable Diffusion Models with Transformers.
- [OpenDiT](https://github.com/NUS-HPC-AI-Lab/OpenDiT): An acceleration for DiT training. We adopt valuable acceleration
  strategies for training progress from OpenDiT.
- [PixArt](https://github.com/PixArt-alpha/PixArt-alpha): An open-source DiT-based text-to-image model.
- [Latte](https://github.com/Vchitect/Latte): An attempt to efficiently train DiT for video.
- [StabilityAI VAE](https://huggingface.co/stabilityai/sd-vae-ft-mse-original): A powerful image VAE model.
- [CLIP](https://github.com/openai/CLIP): A powerful text-image embedding model.
- [T5](https://github.com/google-research/text-to-text-transfer-transformer): A powerful text encoder.
- [LLaVA](https://github.com/haotian-liu/LLaVA): A powerful image captioning model based on [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) and [Yi-34B](https://huggingface.co/01-ai/Yi-34B).
- [PLLaVA](https://github.com/magic-research/PLLaVA): A powerful video captioning model.
- [MiraData](https://github.com/mira-space/MiraData): A large-scale video dataset with long durations and structured caption.

We are grateful for their exceptional work and generous contribution to open source. Special thanks go to the authors of [MiraData](https://github.com/mira-space/MiraData) and [Rectified Flow](https://github.com/gnobitab/RectifiedFlow) for their valuable advice and help. We wish to express gratitude towards AK for sharing this project on social media and Hugging Face for providing free GPU resources for our online Gradio demo.

## Citation

```bibtex
@software{opensora,
  author = {Zangwei Zheng and Xiangyu Peng and Tianji Yang and Chenhui Shen and Shenggui Li and Hongxin Liu and Yukun Zhou and Tianyi Li and Yang You},
  title = {Open-Sora: Democratizing Efficient Video Production for All},
  month = {March},
  year = {2024},
  url = {https://github.com/hpcaitech/Open-Sora}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=hpcaitech/Open-Sora&type=Date)](https://star-history.com/#hpcaitech/Open-Sora&Date)

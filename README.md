

<div align="center">
<h2><font color="red"> Follow-Your-Emoji </font></center> <br> <center>Fine-Controllable and Expressive Freestyle Portrait Animation</h2>

[Yue Ma*](https://mayuelala.github.io/), [Hongyu Liu*](https://kumapowerliu.github.io/), [Hongfa Wang*](https://github.com/mayuelala/FollowYourEmoji), [Heng Pan*](https://github.com/mayuelala/FollowYourEmoji), [Yingqing He](https://github.com/YingqingHe), [Junkun Yuan](https://0-scholar-google-com.brum.beds.ac.uk/citations?user=j3iFVPsAAAAJ&hl=zh-CN),  [Ailing Zeng](https://ailingzeng.site/), [Chengfei Cai](https://github.com/mayuelala/FollowYourEmoji), 
[Heung-Yeung Shum](https://scholar.google.com.hk/citations?user=9akH-n8AAAAJ&hl=en), [Wei Liu](https://scholar.google.com/citations?user=AjxoEpIAAAAJ&hl=zh-CN) and [Qifeng Chen](https://cqf.io)

<a href='https://arxiv.org/abs/2406.01900'><img src='https://img.shields.io/badge/ArXiv-2406.01900-red'></a> <a href='https://github.com/daswer123/FollowYourEmoji-colab/blob/main/README.md'><img src='https://colab.research.google.com/assets/colab-badge.svg'></a> 
<a href='https://follow-your-emoji.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='assets/wechat_group.png'><img src='https://badges.aleen42.com/src/wechat.svg'></a> ![visitors](https://visitor-badge.laobi.icu/badge?page_id=mayuelala.FollowYourEmoji&left_color=green&right_color=red)  [![GitHub](https://img.shields.io/github/stars/mayuelala/FollowYourEmoji?style=social)](https://github.com/mayuelala/FollowYourEmoji,pko) 
</div>

<!-- <table class="center">
  <td><img src="https://follow-your-emoji.github.io/src/teaser/teaser.gif"></td>
  <tr>
    <td align="center" >🤪 For more results, visit our <a href="https://follow-your-emoji.github.io/"><strong>homepage</strong></td>
  <tr>
</td>

</table > -->


## 📣 Updates
- **[2024.07.21]** 🔥 Release `Colab`, thanks for [daswer123](https://github.com/daswer123/FollowYourEmoji-colab/blob/main/README.md)!
- **[2024.07.18]** 🔥 Release `inference code`, `config` and `checkpoints`!
- **[2024.06.07]** 🔥 Release Paper and Project page!

## 🤪 Gallery
<img src="images/index.png" alt="Image 1">

<p>We present <span style="color: #c20557ee">Follow-Your-Emoji</span>, a diffusion-based framework for portrait animation, which animates a reference portrait with target landmark sequences.</p>

## 🤪 Getting Started

### 1. Clone the code and prepare the environment

```bash
pip install -r requirements.txt
```

### 2. Download pretrained weights

[FollowYourEmoji] We also provide our pretrained checkpoints in [Huggingface](https://huggingface.co/YueMafighting/FollowYourEmoji). you could download them and put them into checkpoints folder to inference our model.


### 3. Inference 🚀

```bash
bash infer.sh
```

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.run \
    --nnodes 1 \
    --master_addr $LOCAL_IP \
    --master_port 12345 \
    --node_rank 0 \
    --nproc_per_node 1 \
    infer.py \
    --config ./configs/infer.yaml \
    --model_path /path/to/model \
    --input_path your_own_images_path \
    --lmk_path ./inference_temple/test_temple.npy  \
    --output_path your_own_output_path
```

## 🤪 Make Your Emoji
You can make your own emoji using our model. First, you need to make your emoji temple using MediaPipe. We provide the script in ```make_temple.ipynb```. You can replace the video path with your own emoji video and generate the ```.npy``` file.


```bash
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.run \
    --nnodes 1 \
    --master_addr $LOCAL_IP \
    --master_port 12345 \
    --node_rank 0 \
    --nproc_per_node 1 \
    infer.py \
    --config ./configs/infer.yaml \
    --model_path /path/to/model \
    --input_path your_own_images_path \
    --lmk_path  your_own_temple_path \
    --output_path your_own_output_path
```


## 👨‍👩‍👧‍👦 Follow Family
[Follow-Your-Pose](https://github.com/mayuelala/FollowYourPose): Pose-Guided text-to-Video Generation.

[Follow-Your-Click](https://github.com/mayuelala/FollowYourClick): Open-domain Regional image animation via Short Prompts.

[Follow-Your-Handle](https://github.com/mayuelala/FollowYourHandle): Controllable Video Editing via Control Handle Transformations.

[Follow-Your-Emoji](https://github.com/mayuelala/FollowYourEmoji): Fine-Controllable and Expressive Freestyle Portrait Animation.
  
## Citation 💖
If you find Follow-Your-Emoji useful for your research, welcome to 🌟 this repo and cite our work using the following BibTeX:
```bibtex
@article{ma2024follow,
  title={Follow-Your-Emoji: Fine-Controllable and Expressive Freestyle Portrait Animation},
  author={Ma, Yue and Liu, Hongyu and Wang, Hongfa and Pan, Heng and He, Yingqing and Yuan, Junkun and Zeng, Ailing and Cai, Chengfei and Shum, Heung-Yeung and Liu, Wei and others},
  journal={arXiv preprint arXiv:2406.01900},
  year={2024}
}
```

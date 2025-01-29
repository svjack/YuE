<p align="center">
    <img src="./assets/logo/白底.png" width="400" />
</p>

```bash
sudo apt-get update && sudo apt-get install cbm ffmpeg git-lfs

conda create -n yue python=3.10 # Python >=3.8 is recommended.
conda activate yue
pip install ipykernel
python -m ipykernel install --user --name yue --display-name "yue"

# install cuda >= 11.8
#conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia

git clone https://huggingface.co/spaces/svjack/YuE && cd YuE
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

cd inference
python infer.py \
    --stage1_model m-a-p/YuE-s1-7B-anneal-en-cot \
    --stage2_model m-a-p/YuE-s2-1B-general \
    --genre_txt genre.txt \
    --lyrics_txt lyrics.txt \
    --run_n_segments 2 \
    --stage2_batch_size 4 \
    --output_dir ./output \
    --cuda_idx 0 \
    --max_new_tokens 3000

# ffmpeg -i output/cot_inspiring-female-uplifting-pop-airy-vocal-electronic-bright-vocal-vocal_tp0@93_T100_rp102_maxtk3000_mixed_f5e0c515-f10c-4a13-a47d-76c6cc1712fd.mp3' -c:v libx264 -c:a aac -strict experimental output/cot_inspiring-female-uplifting-pop-airy-vocal-electronic-bright-vocal-vocal_tp0@93__T1@0_rp1@2_maxtk3000_mixed_{5e0c515-f10c-4a13-a47d-76c6cc1712fd.mp4
```


https://github.com/user-attachments/assets/30f53d32-efc3-4dde-8d58-68ecf9c8ec19


```bash
python infer.py \
    --stage1_model m-a-p/YuE-s1-7B-anneal-zh-cot \
    --stage2_model m-a-p/YuE-s2-1B-general \
    --genre_txt genre.txt \
    --lyrics_txt lyrics_zh.txt \
    --run_n_segments 2 \
    --stage2_batch_size 4 \
    --output_dir ./zh_output \
    --cuda_idx 0 \
    --max_new_tokens 3000

python infer.py \
    --stage1_model m-a-p/YuE-s1-7B-anneal-zh-cot \
    --stage2_model m-a-p/YuE-s2-1B-general \
    --genre_txt genre.txt \
    --lyrics_txt lyrics_zh.txt \
    --run_n_segments 2 \
    --stage2_batch_size 4 \
    --output_dir ./zh_output_new \
    --cuda_idx 0 \
    --max_new_tokens 3000 \
    --prompt_end_time 360

# ffmpeg -i cot_zh_demo.mp3-codec:a libmp3lame -b:a 192k -f mp4 -vn cot_zh_demo.mp4
```

```txt
[verse]
注目夕阳晚，彩霞满天边
思念绕心间，无法抗拒的情感
明知曾令你失望，过错在眼前
此刻归来只为，修补水痕涟涟

[chorus]
无论你选择何路，我在身后紧守护
追逐你的梦想，握住希望与光
此情此意此刻，难以抵拒
我已决绝，不应退缩
深知你无法抵抗，我不会退缩

[verse]
有人笑我愚，为爱独自追逐
这般的深情，唯你我心照不误
心之所向，唯你一人
不愿你远去，我早已心有所属
```



https://github.com/user-attachments/assets/42822b4d-2f7d-4380-8ecf-4d46d3c99dfb





https://github.com/user-attachments/assets/137e2bd0-7812-471d-bffd-da863a0234a4


```bash
cd ..
python app.py
```

<p align="center">
    <a href="https://map-yue.github.io/">Demo 🎶</a> &nbsp;|&nbsp; 📑 <a href="">Paper (coming soon)</a>
    <br>
    <a href="https://huggingface.co/m-a-p/YuE-s1-7B-anneal-en-cot">YuE-s1-7B-anneal-en-cot 🤗</a> &nbsp;|&nbsp; <a href="https://huggingface.co/m-a-p/YuE-s1-7B-anneal-en-icl">YuE-s1-7B-anneal-en-icl 🤗</a> &nbsp;|&nbsp; <a href="https://huggingface.co/m-a-p/YuE-s1-7B-anneal-jp-kr-cot">YuE-s1-7B-anneal-jp-kr-cot 🤗</a>
    <br>
    <a href="https://huggingface.co/m-a-p/YuE-s1-7B-anneal-jp-kr-icl">YuE-s1-7B-anneal-jp-kr-icl 🤗</a> &nbsp;|&nbsp; <a href="https://huggingface.co/m-a-p/YuE-s1-7B-anneal-zh-cot">YuE-s1-7B-anneal-zh-cot 🤗</a> &nbsp;|&nbsp; <a href="https://huggingface.co/m-a-p/YuE-s1-7B-anneal-zh-icl">YuE-s1-7B-anneal-zh-icl 🤗</a>
    <br>
    <a href="https://huggingface.co/m-a-p/YuE-s2-1B-general">YuE-s2-1B-general 🤗</a> &nbsp;|&nbsp; <a href="https://huggingface.co/m-a-p/YuE-upsampler">YuE-upsampler 🤗</a>
</p>

---
Our model's name is **YuE (乐)**. In Chinese, the word means "music" and "happiness." Some of you may find words that start with Yu hard to pronounce. If so, you can just call it "yeah." We wrote a song with our model's name, see [here](assets/logo/yue.mp3).

YuE is a groundbreaking series of open-source foundation models designed for music generation, specifically for transforming lyrics into full songs (lyrics2song). It can generate a complete song, lasting several minutes, that includes both a catchy vocal track and accompaniment track. YuE is capable of modeling diverse genres/languages/vocal techniques. Please visit the [**Demo Page**](https://map-yue.github.io/) for amazing vocal performance.

## News and Updates

* **2025.01.29 🎉**: We have updated the license description. we **ENCOURAGE** artists and content creators to sample and incorporate outputs generated by our model into their own works, and even monetize them. The only requirement is to credit our name: **YuE by M-A-P**.
* **2025.01.28 🫶**: Thanks to Fahd for creating a tutorial on how to quickly get started with YuE. Here is his [demonstration](https://www.youtube.com/watch?v=RSMNH9GitbA).
* **2025.01.26 🔥**: We have released the **YuE** series.

<br>

---
## TODOs
- [ ] Support dual-track ICL mode.
- [ ] Support gradio interface.
- [ ] Support transformers tensor parallel.
- [ ] Online serving on huggingface space.
- [ ] Example finetune code for enabling BPM control using 🤗 Transformers.

---

## Hardware and Performance

### **GPU Memory**
YuE requires significant GPU memory for generating long sequences. Below are the recommended configurations:

- **For GPUs with 24GB memory or less**: Run **up to 2 sessions** concurrently to avoid out-of-memory (OOM) errors.
- **For full song generation** (many sessions, e.g., 4 or more): Use **GPUs with at least 80GB memory**. i.e. H800, A100, or multiple RTX4090s with tensor parallel.

To customize the number of sessions, the interface allows you to specify the desired session count. By default, the model runs **2 sessions** (1 verse + 1 chorus) to avoid OOM issue.

### **Execution Time**
On an **H800 GPU**, generating 30s audio takes **150 seconds**.
On an **RTX 4090 GPU**, generating 30s audio takes approximately **360 seconds**. 

---

## Quickstart
Quick start **VIDEO TUTORIAL** by Fahd: [Link here](https://www.youtube.com/watch?v=RSMNH9GitbA). We recommend watching this video if you are not familiar with machine learning or the command line.

### 1. Install environment and dependencies
Make sure properly install flash attention 2 to reduce VRAM usage. 
```bash
# We recommend using conda to create a new environment.
conda create -n yue python=3.8 # Python >=3.8 is recommended.
conda activate yue
# install cuda >= 11.8
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia
pip install -r requirements.txt

# For saving GPU memory, FlashAttention 2 is mandatory. 
# Without it, long audio may lead to out-of-memory (OOM) errors.
# Be careful about matching the cuda version and flash-attn version
pip install flash-attn --no-build-isolation
```

### 2. Download the infer code and tokenizer
```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://github.com/multimodal-art-projection/YuE.git

cd YuE/inference/
git clone https://huggingface.co/m-a-p/xcodec_mini_infer
```

### 3. Run the inference
Now generate music with **YuE** using 🤗 Transformers. Make sure your step [1](#1-install-environment-and-dependencies) and [2](#2-download-the-infer-code-and-tokenizer) are properly set up. 

Note:
- Set `--run_n_segments` to the number of lyric sections if you want to generate a full song. Additionally, you can increase `--stage2_batch_size` based on your available GPU memory.

- You may customize the prompt in `genre.txt` and `lyrics.txt`. See prompt engineering guide [here](#prompt-engineering-guide).

- LM ckpts will be automatically downloaded from huggingface. 


```bash
# This is the CoT mode.
cd YuE/inference/
python infer.py \
    --stage1_model m-a-p/YuE-s1-7B-anneal-en-cot \
    --stage2_model m-a-p/YuE-s2-1B-general \
    --genre_txt genre.txt \
    --lyrics_txt lyrics.txt \
    --run_n_segments 2 \
    --stage2_batch_size 4 \
    --output_dir ./output \
    --cuda_idx 0 \
    --max_new_tokens 3000 
```

If you want to use music in-context-learning (provide a reference song), enable `--use_audio_prompt`, `--prompt_start_time`, and `--prompt_end_time` to specify the audio segment. 

Note: 
- ICL requires a different ckpt, e.g. `m-a-p/YuE-s1-7B-anneal-en-icl`.

- Music ICL generally requires a 30s audio segment. The model will write new songs with similiar style of the provided audio, and may improve musicality.

- We have 4 modes for ICL: mix, vocal, instrumental, and dual-track. 

- We currently only support mix mode. 

- Dual-track mode work the best, will support in the infer code soon.

```bash
# This is the ICL mode. Currently only mix-ICL is supported.
cd YuE/inference/
python infer.py \
    --stage1_model m-a-p/YuE-s1-7B-anneal-en-icl \
    --stage2_model m-a-p/YuE-s2-1B-general \
    --genre_txt genre.txt \
    --lyrics_txt lyrics.txt \
    --run_n_segments 2 \
    --stage2_batch_size 4 \
    --output_dir ./output \
    --cuda_idx 0 \
    --max_new_tokens 3000 \
    --audio_prompt_path {YOUR_AUDIO_FILE} \
    --prompt_start_time 0 \
    --prompt_end_time 30 
```
---
 
## Prompt Engineering Guide
The prompt consists of three parts: genre tags, lyrics, and ref audio.

### Genre Tagging Prompt
1. An example genre tagging prompt can be found [here](inference/prompt_examples/genre.txt).

2. A stable tagging prompt usually consists of five components: genre, instrument, mood, gender, and timbre. All five should be included if possible, separated by space (space delimiter).

3. Although our tags have an open vocabulary, we have provided the top 200 most commonly used [tags](./top_200_tags.json). It is recommended to select tags from this list for more stable results.

3. The order of the tags is flexible. For example, a stable genre tagging prompt might look like: "inspiring female uplifting pop airy vocal electronic bright vocal vocal."

4. Additionally, we have introduced the "Mandarin" and "Cantonese" tags to distinguish between Mandarin and Cantonese, as their lyrics often share similarities.

### Lyrics Prompt
1. An example lyric prompt can be found [here](inference/prompt_examples/lyrics.txt).

2. We support multiple languages, including but not limited to English, Mandarin Chinese, Cantonese, Japanese, and Korean. The default top language distribution during the annealing phase is revealed in [issue 12](https://github.com/multimodal-art-projection/YuE/issues/12#issuecomment-2620845772). A language ID on a specific annealing checkpoint indicates that we have adjusted the mixing ratio to enhance support for that language.

3. The lyrics prompt should be divided into sessions, with structure labels (e.g., [verse], [chorus], [bridge], [outro]) prepended. Each session should be separated by 2 newline character "\n\n".

4. **DONOT** put too many words in a single segment, since each session is around 30s (`--max_new_tokens 3000` by default).

5. We find that [intro] label is less stable, so we recommend starting with [verse] or [chorus].

6. For generating music with no vocal, see [issue 18](https://github.com/multimodal-art-projection/YuE/issues/18).


### Audio Prompt
1. Audio prompt is optional. Providing ref audio for ICL usually increase the good case rate, and result in less diversity since the generated token space is bounded by the ref audio. CoT only (no ref) will result in a more diverse output.

1. We find that dual-track ICL mode gives the best musicality and prompt following. We will support this mode soon.

2. Use the chorus part of the music as prompt will result in better musicality.

---

## License Agreement \& Disclaimer  
- Our models are licensed under Creative Commons Attribution Non Commercial 4.0, meaning the model weights themselves **CANNOT** be used for commercial purposes.
- However, we **ENCOURAGE** artists and content creators to sample and incorporate outputs generated by our model into their own works, and even monetize them. The only requirement is to credit our name: **YuE by M-A-P**.
- We **DO NOT assume any responsibility** for any misuse of this model, including but not limited to **illegal, malicious, or unethical activities**.  
- Users are solely responsible for any content generated with the model and any consequences arising from its use.  

---

## Citation

If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil: :)

```BibTeX
@misc{yuan2025yue,
  title={YuE: Open Music Foundation Models for Full-Song Generation},
  author={Ruibin Yuan and Hanfeng Lin and Shawn Guo and Ge Zhang and Jiahao Pan and Yongyi Zang and Haohe Liu and Xingjian Du and Xeron Du and Zhen Ye and Tianyu Zheng and Yinghao Ma and Minghao Liu and Lijun Yu and Zeyue Tian and Ziya Zhou and Liumeng Xue and Xingwei Qu and Yizhi Li and Tianhao Shen and Ziyang Ma and Shangda Wu and Jun Zhan and Chunhui Wang and Yatian Wang and Xiaohuan Zhou and Xiaowei Chi and Xinyue Zhang and Zhenzhu Yang and Yiming Liang and Xiangzhou Wang and Shansong Liu and Lingrui Mei and Peng Li and Yong Chen and Chenghua Lin and Xie Chen and Gus Xia and Zhaoxiang Zhang and Chao Zhang and Wenhu Chen and Xinyu Zhou and Xipeng Qiu and Roger Dannenberg and Jiaheng Liu and Jian Yang and Stephen Huang and Wei Xue and Xu Tan and Yike Guo}, 
  howpublished={\url{https://github.com/multimodal-art-projection/YuE}},
  year={2025},
  note={GitHub repository}
}
```
<br>

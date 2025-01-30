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

python app.py
### Ctrl + C

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

python infer.py \
    --stage1_model m-a-p/YuE-s1-7B-anneal-zh-cot \
    --stage2_model m-a-p/YuE-s2-1B-general \
    --genre_txt genre.txt \
    --lyrics_txt lyrics_zh.txt \
    --run_n_segments 20 \
    --stage2_batch_size 4 \
    --output_dir ./zh_output_long \
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




https://github.com/user-attachments/assets/300e0554-551f-4ed7-9dad-0cbf48ca9d54



```python
# 定义词数据
lyrics_data = {
    "苏轼_水调歌头": """[verse]
明月几时有？把酒问青天。
不知天上宫阙，今夕是何年。
我欲乘风归去，又恐琼楼玉宇，高处不胜寒。
起舞弄清影，何似在人间。

[chorus]
不应有恨，何事长向别时圆？
人有悲欢离合，月有阴晴圆缺，此事古难全。
但愿人长久，千里共婵娟.

[outro]
""",

    "辛弃疾_青玉案": """[verse]
东风夜放花千树，更吹落、星如雨。
宝马雕车香满路。
凤箫声动，玉壶光转，一夜鱼龙舞。

[chorus]
蛾儿雪柳黄金缕，笑语盈盈暗香去。
众里寻他千百度，蓦然回首，那人却在，灯火阑珊处.

[outro]""",

    "李清照_声声慢": """[verse]
寻寻觅觅，冷冷清清，凄凄惨惨戚戚。
乍暖还寒时候，最难将息。
三杯两盏淡酒，怎敌他、晚来风急！

[chorus]
满地黄花堆积，憔悴损，如今有谁堪摘？
守着窗儿，独自怎生得黑！
梧桐更兼细雨，到黄昏、点点滴滴.

[outro]""",

    "岳飞_满江红": """[verse]
怒发冲冠，凭栏处、潇潇雨歇。
抬望眼、仰天长啸，壮怀激烈。
三十功名尘与土，八千里路云和月。

[chorus]
靖康耻，犹未雪。
驱长车踏破，贺兰山缺。
壮志饥餐胡虏肉，笑谈渴饮匈奴血。
待从头、收拾旧山河，朝天阙！

[outro]""",

    "苏轼_念奴娇": """[verse]
大江东去，浪淘尽，千古风流人物。
故垒西边，人道是，三国周郎赤壁。
乱石穿空，惊涛拍岸，卷起千堆雪。

[chorus]
遥想公瑾当年，小乔初嫁了，雄姿英发。
羽扇纶巾，谈笑间，樯橹灰飞烟灭。
故国神游，多情应笑我，早生华发.

[outro]""",

    "辛弃疾_永遇乐": """[verse]
千古江山，英雄无觅孙仲谋处。
舞榭歌台，风流总被雨打风吹去。
斜阳草树，寻常巷陌，人道寄奴曾住。

[chorus]
想当年，金戈铁马，气吞万里如虎。
四十三年，望中犹记，烽火扬州路。
可堪回首，佛狸祠下，一片神鸦社鼓.

[outro]""",

    "辛弃疾_破阵子": """[verse]
醉里挑灯看剑，梦回吹角连营。
八百里分麾下炙，五十弦翻塞外声。

[chorus]
沙场秋点兵。
了却君王天下事，赢得生前身后名。
可怜白发生！

[outro]""",

    "李清照_如梦令": """[verse]
常记溪亭日暮，沉醉不知归路。
兴尽晚回舟，误入藕花深处。

[chorus]
争渡，争渡，惊起一滩鸥鹭.

[outro]""",

    "温庭筠_菩萨蛮": """[verse]
小山重叠金明灭，鬓云欲度香腮雪。
懒起画蛾眉，弄妆梳洗迟。

[chorus]
照花前后镜，花面交相映。
新帖绣罗襦，双双金鹧鸪.

[outro]""",

    "杨慎_临江仙": """[verse]
滚滚长江东逝水，浪花淘尽英雄。
是非成败转头空。
青山依旧在，几度夕阳红。

[chorus]
白发渔樵江渚上，惯看秋月春风。
一壶浊酒喜相逢。
古今多少事，都付笑谈中。

[outro]"""
}

lyrics_data = {
    "苏轼_水调歌头": """[verse]
明月几时有？把酒问青天。
不知天上宫阙，今夕是何年。
我欲乘风归去，又恐琼楼玉宇，高处不胜寒。
起舞弄清影，何似在人间。
不应有恨，何事长向别时圆？
人有悲欢离合，月有阴晴圆缺，此事古难全。
但愿人长久，千里共婵娟.

[chorus]

[outro]
""",

    "辛弃疾_青玉案": """[verse]
东风夜放花千树，更吹落、星如雨。
宝马雕车香满路。
凤箫声动，玉壶光转，一夜鱼龙舞。
蛾儿雪柳黄金缕，笑语盈盈暗香去。
众里寻他千百度，蓦然回首，那人却在，灯火阑珊处.

[chorus]

[outro]""",

    "李清照_声声慢": """[verse]
寻寻觅觅，冷冷清清，凄凄惨惨戚戚。
乍暖还寒时候，最难将息。
三杯两盏淡酒，怎敌他、晚来风急！
满地黄花堆积，憔悴损，如今有谁堪摘？
守着窗儿，独自怎生得黑！
梧桐更兼细雨，到黄昏、点点滴滴.

[chorus]

[outro]""",

    "岳飞_满江红": """[verse]
怒发冲冠，凭栏处、潇潇雨歇。
抬望眼、仰天长啸，壮怀激烈。
三十功名尘与土，八千里路云和月。
靖康耻，犹未雪。
驱长车踏破，贺兰山缺。
壮志饥餐胡虏肉，笑谈渴饮匈奴血。
待从头、收拾旧山河，朝天阙！

[chorus]

[outro]""",

    "苏轼_念奴娇": """[verse]
大江东去，浪淘尽，千古风流人物。
故垒西边，人道是，三国周郎赤壁。
乱石穿空，惊涛拍岸，卷起千堆雪。
遥想公瑾当年，小乔初嫁了，雄姿英发。
羽扇纶巾，谈笑间，樯橹灰飞烟灭。
故国神游，多情应笑我，早生华发.

[chorus]

[outro]""",

    "辛弃疾_永遇乐": """[verse]
千古江山，英雄无觅孙仲谋处。
舞榭歌台，风流总被雨打风吹去。
斜阳草树，寻常巷陌，人道寄奴曾住。
想当年，金戈铁马，气吞万里如虎。
四十三年，望中犹记，烽火扬州路。
可堪回首，佛狸祠下，一片神鸦社鼓.

[chorus]

[outro]""",

    "辛弃疾_破阵子": """[verse]
醉里挑灯看剑，梦回吹角连营。
八百里分麾下炙，五十弦翻塞外声。
沙场秋点兵。
了却君王天下事，赢得生前身后名。
可怜白发生！

[chorus]

[outro]""",

    "李清照_如梦令": """[verse]
常记溪亭日暮，沉醉不知归路。
兴尽晚回舟，误入藕花深处。
争渡，争渡，惊起一滩鸥鹭.

[chorus]

[outro]""",

    "温庭筠_菩萨蛮": """[verse]
小山重叠金明灭，鬓云欲度香腮雪。
懒起画蛾眉，弄妆梳洗迟。
照花前后镜，花面交相映。
新帖绣罗襦，双双金鹧鸪.

[chorus]

[outro]""",

    "杨慎_临江仙": """[verse]
滚滚长江东逝水，浪花淘尽英雄。
是非成败转头空。
青山依旧在，几度夕阳红。
白发渔樵江渚上，惯看秋月春风。
一壶浊酒喜相逢。
古今多少事，都付笑谈中。

[chorus]

[outro]"""
}

# 将内容保存到文件
for filename, content in lyrics_data.items():
    # 创建文件名
    file_name = f"lyrics_{filename}.txt"
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(content)
    print(f"文件已生成: {file_name}")

print("所有文件生成完成！")

```

```bash
#!/bin/bash

# 创建输出目录
mkdir -p ./zh_output_song_short

# 遍历 lyrics_*.txt 文件
for lyrics_file in lyrics_*.txt; do
    # 复制当前 lyrics 文件到 lyrics_zh.txt
    cp "$lyrics_file" lyrics_zh.txt
    
    # 提取文件名（不含扩展名），用于命名输出目录
    output_dir_name=$(basename "$lyrics_file" .txt)
    output_dir=./zh_output_song_short/"$output_dir_name"
    
    echo "Processing $lyrics_file..."
    echo "Output directory: $output_dir"
    
    # 创建输出目录
    mkdir -p "$output_dir"
    
    python infer.py \
        --stage1_model m-a-p/YuE-s1-7B-anneal-zh-cot \
        --stage2_model m-a-p/YuE-s2-1B-general \
        --genre_txt genre.txt \
        --lyrics_txt lyrics_zh.txt \
        --run_n_segments 4 \
        --stage2_batch_size 4 \
        --output_dir "$output_dir" \
        --cuda_idx 0 \
        --max_new_tokens 3000 \
        --prompt_end_time 3600 \
        
done

echo "Processing completed!"
```

```python
import os
from datasets import Dataset, Audio

# 定义歌词文件路径和音乐文件输出路径
lyrics_dir = "."  # 歌词文件所在的目录
music_dir = "./zh_output_song_short"  # 音乐文件所在的目录

# 初始化数据集列表
data = []

# 遍历歌词文件
for lyrics_file in os.listdir(lyrics_dir):
    if lyrics_file.startswith("lyrics_") and lyrics_file.endswith(".txt") and "zh" not in lyrics_file:
        # 提取作者和词牌名
        filename = lyrics_file[len("lyrics_"):-len(".txt")]
        author, title = filename.split("_", 1)
        
        # 读取歌词内容
        with open(os.path.join(lyrics_dir, lyrics_file), "r", encoding="utf-8") as f:
            lyrics_content = f.read()
        
        # 构建音乐文件路径
        music_subdir = os.path.join(music_dir, f"lyrics_{filename}")
        
        # 检查音乐子目录是否存在
        if os.path.exists(music_subdir):
            # 查找音乐文件
            music_files = [f for f in os.listdir(music_subdir) if f.endswith(".wav") or f.endswith(".mp3")]
            
            # 如果找到音乐文件，则添加到数据集
            if music_files:
                music_file = os.path.join(music_subdir, music_files[0])
                # 使用 Audio 类型保存音频文件路径
                #music_content = {"path": music_file, "array": None, "sampling_rate": None}
                
                # 添加到数据集
                data.append({
                    "author": author,
                    "title": title,
                    "lyrics": lyrics_content,
                    "music": music_file
                })
        else:
            print(f"音乐子目录不存在: {music_subdir}")

# 创建 Hugging Face Dataset
if data:  # 确保数据列表不为空
    dataset = Dataset.from_dict({
        "author": [item["author"] for item in data],
        "title": [item["title"] for item in data],
        "lyrics": [item["lyrics"] for item in data],
        "music": [item["music"] for item in data]
    })

    # 将 music 列转换为 Audio 类型
    dataset = dataset.cast_column("music", Audio())

    # 保存数据集到磁盘
    dataset.save_to_disk("./lyrics_music_dataset")
    print("数据集已保存到 ./lyrics_music_dataset")
else:
    print("没有有效的数据可以生成数据集。")

dataset.push_to_hub("svjack/YuE-Song-Ci-9")
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

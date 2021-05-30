# LayoutLM

## Fine-tuning Example

We evaluate LayoutLM on several document image understanding datasets, and it outperforms several SOTA pre-trained models and approaches.

Setup environment as follows:

~~~bash
conda create -n layoutlm python=3.6
conda activate layoutlm
conda install pytorch==1.4.0 cudatoolkit=10.1 -c pytorch
git clone https://github.com/NVIDIA/apex && cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
pip install .
## For development mode
# pip install -e ".[dev]"
~~~

### Sequence Labeling Task


We give a fine-tuning example for sequence labeling tasks. You can run this example on [FUNSD](https://guillaumejaume.github.io/FUNSD/), a dataset for document understanding tasks.

First, we need to preprocess the JSON file into txt. You can run the preprocessing scripts `funsd_preprocess.py` in the `scripts` directory. For more options, please refer to the arguments.

~~~bash
cd examples/seq_labeling
./preprocess.sh
~~~

After preprocessing, run LayoutLM as follows:

~~~bash
python run_seq_labeling.py  --data_dir data \
                            --model_type layoutlm \
                            --model_name_or_path path/to/pretrained/model/directory \
                            --do_lower_case \
                            --max_seq_length 512 \
                            --do_train \
                            --num_train_epochs 100.0 \
                            --logging_steps 10 \
                            --save_steps -1 \
                            --output_dir path/to/output/directory \
                            --labels data/labels.txt \
                            --per_gpu_train_batch_size 16 \
                            --per_gpu_eval_batch_size 16 \
                            --fp16
~~~

### Prediction on a new Image

Import the module custom_preprocess.py <br>
Pass the image to custom_img_annotation_.write_annoteFile() for preprocessing the new image. <br>
Calling custom_img_annotation_.convert() and custom_img_annotation_.seg() will proce the test.txt file required by layoutLm model for prediction. <br>

After the preprocessing, run layoutlm using --dopredict method as follows. <br>

~~~bash
python run_seq_labeling.py  --do_predict \
                            --data_dir data \
                            --model_type layoutlm \
                            --model_name_or_path output \
                            --do_lower_case \
                            --output_dir output \
                            --labels data/labels.txt \
                            --fp16
~~~
### Results

#### SROIE (field-level)


| Model                                                        | Hmean      |
| ------------------------------------------------------------ | ---------- |
| BERT-Large                                                   | 90.99%     |
| RoBERTa-Large                                                | 92.80%     |
| [Ranking 1st in SROIE](https://rrc.cvc.uab.es/?ch=13&com=evaluation&task=3) | 94.02%     |
| [**LayoutLM**](https://rrc.cvc.uab.es/?ch=13&com=evaluation&view=method_info&task=3&m=71448) | **96.04%** |

#### RVL-CDIP

| Model                                                        | Accuracy   |
| ------------------------------------------------------------ | ---------- |
| BERT-Large                                                   | 89.92%     |
| RoBERTa-Large                                                | 90.11%     |
| [VGG-16 (Afzal et al., 2017)](https://arxiv.org/abs/1704.03557) | 90.97%     |
| [Stacked CNN Ensemble (Das et al., 2018)](https://arxiv.org/abs/1801.09321) | 92.21%     |
| [LadderNet (Sarkhel & Nandi, 2019)](https://www.ijcai.org/Proceedings/2019/0466.pdf) | 92.77%     |
| [Multimodal Ensemble (Dauphinee et al., 2019)](https://arxiv.org/abs/1912.04376) | 93.07%     |
| **LayoutLM**                                                 | **94.42%** |

#### FUNSD (field-level)

| Model         | Precision  | Recall     | F1         |
| ------------- | ---------- | ---------- | ---------- |
| BERT-Large    | 0.6113     | 0.7085     | 0.6563     |
| RoBERTa-Large | 0.6780     | 0.7391     | 0.7072     |
| **LayoutLM**  | **0.7677** | **0.8195** | **0.7927** |

## Citation

If you find LayoutLM useful in your research, please cite the following paper:

``` latex
@misc{xu2019layoutlm,
    title={LayoutLM: Pre-training of Text and Layout for Document Image Understanding},
    author={Yiheng Xu and Minghao Li and Lei Cui and Shaohan Huang and Furu Wei and Ming Zhou},
    year={2019},
    eprint={1912.13318},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## License

This project is licensed under the license found in the LICENSE file in the root directory of this source tree.
Portions of the source code are based on the [transformers](https://github.com/huggingface/transformers) project.
[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information

For help or issues using LayoutLM, please submit a GitHub issue.

For other communications related to LayoutLM, please contact Lei Cui (`lecu@microsoft.com`), Furu Wei (`fuwei@microsoft.com`).


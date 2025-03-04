# [EMNLP 2024] CmdCaliper: A Semantic-Aware Command-Line Embedding Model and Dataset for Security Research
## [[Dataset](https://huggingface.co/datasets/CyCraftAI/CyPHER)] [[Code](https://github.com/cycraft-corp/CmdCaliper)] [[Paper](https://arxiv.org/abs/2411.01176)] 
## TL;DR
We create the first similar command lines dataset generated by large language models and introduce an efficient command-line embedding model CmdCaliper, surpassing current models in performance, with all data and model weights publicly released.

## Abstract
This research addresses command-line embedding in cybersecurity, a field obstructed by the lack of comprehensive datasets due to privacy and regulation concerns. We propose the first dataset of similar command lines, named CyPHER, for training and unbiased evaluation. The training set is generated using a set of large language models (LLMs) comprising 28,520 similar command-line pairs. Our testing dataset consists of 2,807 similar command-line pairs sourced from authentic command-line data.

In addition, we propose a command-line embedding model named CmdCaliper, enabling the computation of semantic similarity with command lines. Performance evaluations demonstrate that the smallest version of CmdCaliper (30 million parameters) suppresses state-of-the-art (SOTA) sentence embedding models with ten times more parameters across various tasks (e.g., malicious command-line detection and similar command-line retrieval).

Our study explores the feasibility of data generation using LLMs in the cybersecurity domain. Furthermore, we release our proposed command-line dataset, embedding models’ weights and all program codes to the public. This advancement paves the way for more effective command-line embedding for future researchers.

## Requirements
```
conda create -yn cmdcaliper python=3.10
conda activate cmdcaliper

pip install -r requirements.txt
```

## Similar Command-Line Pairs Synthesis Pipeline
### Step1: Credential Configurations of LLM Pool
- Copy the template configuration file to create your own configuration file first
```
cp credential_config_template.yaml credential_config.yaml
```
- Please configure your `llm_pool_info` in the `credential_config.yaml` file, including specifying the inference `engine_name`, `model_name` and the corresponding `api_key` and `base_url`.
- Below is an example of using both gpt-4o and gpt-4o-mini in the LLM pool:
```
# credential_config.yaml
llm_pool_info:
    - engine_name: "OpenAIInferenceEngine"
      model_name: "gpt-4o-mini"
      engine_arguments:
        api_key: [YOU OPENAI APIKEY]
    - engine_name: "OpenAIInferenceEngine"
      model_name: "gpt-4o"
      engine_arguments:
        api_key: [YOU OPENAI APIKEY]
```
- The currently supported engines are `OpenAIInferenceEngine`, `GoogleInferenceEngine`, and `AnthropicInferenceEngine`. (Please refer to `./src/inferencer.py` for more details.)

### Step2: Single Command Line Synthesis
```
python3 synthesized_cmds.py \
    --path-to-seed-cmds [PATH TO SEED CMD]
    --path-to-output-dir [PATH TO OUTPUT DIR] \
    --max-generation-num [DATA NUM TO GENERATE] \
    --path-to-credential-config ./credential_config.yaml
```

- `--path-to-seed-cmds`: Path to the initial seed commands file. This JSON file contains the starting data required for the synthesis process.
- `--path-to-output-dir`: Directory where the generated data and logs will be stored.
- `--max-generation-num`: The total number of data items to generate.
- `--path-to-credential-config`: Path to the credential configuration file. This YAML file includes settings for the LLM pool and API key information.

### Step3: Positive Command Line Synthesis
```
python3 synthesize_positive_cmds.py \
    --path-to-all-cmds [PATH TO ALL CMDS] \
    --path-to-output-dir [PATH TO OUTPUT DIR] \
    --path-to-credential-config ./credential_config.yaml
```
- `--path-to-all-cmds`: Path to a file containing command lines for which you want to generate similar commands.
- `--path-to-output-dir`: Directory where the generated data and logs will be stored.
- `--path-to-credential-config`: Path to the credential configuration file. This YAML file includes settings for the LLM pool and API key information.

## CmdCaliper: A Semantic-Aware Command-Line Embedding Model

### Model Zoo of CmdCaliper


|       Model       | Params (B) | MRR@10 | Top@10 |
|:-----------------:|:----------:|:------:|:-------:|
| [CmdCaliper-Small](https://huggingface.co/CyCraftAI/CmdCaliper-small)|    0.03   |  87.78 | 94.76 | 
| [CmdCaliper-Base](https://huggingface.co/CyCraftAI/CmdCaliper-base)|    0.11   |  88.47 | 95.26 | 
| [CmdCaliper-Large](https://huggingface.co/CyCraftAI/CmdCaliper-large) |    0.335   |  89.9 | 95.65 |


### Evaluation on [CyPHER](https://huggingface.co/datasets/CyCraftAI/CyPHER)
- To reproduce the performance on the testing set of CyPHER as reported in the paper, you can evaluate different models using the following command:

```
python3 evaluate.py --model-name [MODEL_NAME] \
    --batch-size 16 --device cuda
```
- Replace `[MODEL_NAME]` with one of the following options to evaluate the respective model:
    - `"CyCraftAI/CmdCaliper-small"`
    - `"CyCraftAI/CmdCaliper-base"`
    - `"CyCraftAI/CmdCaliper-large"`
    - `"thenlper/gte-small"`
- Adjust the `--batch-size` parameter if necessary to accommodate hardware constraints.

## Training Scripts of CmdCaliper
We provide the training scripts with the configs of CmdCaliper reported in our paper. 

### Training Command
```
python3 train.py \
    --temperature 0.05 \
    --lr 0.00002 \
    --path-to-checkpoint-dir ./checkpoints \
    --path-to-train-data-dir ./data/train_data \
    --path_to_eval_data_dir ./data/eval_data \
    --path-to-model-weight thenlper/gte-small \
    --epochs 2
```

### Data Preparation
You need to prepare a `data.json` file for both your training and evaluation datasets. Place these files in the directories specified by `--path-to-train-data-dir` and `--path-to-eval-data-dir`. In our paper, we extracted 1,000 command line pairs from the training data to serve as the evaluation dataset.

Please make sure the data in `data.json` follow this format:
```
[
  [cmd1, positive_cmd1],
  [cmd2, positive_cmd2],
  [cmd3, positive_cmd3],
  [cmd4, positive_cmd4],
  ...
]
```

#### Automatic Evaluation Split

You can also automatically split your training data into training and evaluation datasets by using the `--train-percentage` argument. Note that this will result in a different evaluation dataset for each training session.

## Checkpoints

During training, the following will be saved in the directory specified by `--path-to-checkpoint-dir`:

- Model weights
- Optimizer state
- Learning rate scheduler state

These files allow you to resume training if needed. Additionally, a `huggingface_model` directory will be created, containing the model weights in Transformers style.



## Citation
```
@inproceedings{huang2024cmdcaliper,
  title={{C}md{C}aliper: A Semantic-Aware Command-Line Embedding Model and Dataset for Security Research},
  author={Sian-Yao Huang and Cheng-Lin Yang and Che-Yu Lin and Chun-Ying Huang},
  booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
  pages = "20188--20206",
  month = nov,
  year={2024}
} 
```

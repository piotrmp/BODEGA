# BODEGA: Benchmark for Adversarial Example Generation in Credibility Assessment

This repository contains the BODEGA benchmark for evaluating the robustness of text classifiers, i.e. their ability to
maintain the correct prediction for test examples that were modified by a malicious attacker. BODEGA is using tasks
related to the detection of misinformation and aims to simulate the real usecase of social media platforms
employing ML classifiers for content filtering. The basic tasks (with IDs) are:

- Style-based news bias assessment (HN),
- Propaganda detection (PR2),
- Fact checking (FC),
- Rumour detection (RD).

The victim classifiers include:

- BiLSTM
- fine-tuned BERT.

The full description of the benchmark is available in the article preprint on
arXiv ([Verifying the Robustness of Automatic Credibility Assessment](https://arxiv.org/abs/2303.08032)). Apart from
background information
(related work, motivation, description of the tasks, explanation of evaluation), it includes the results of attacks
involving common adversarial example generation strategies and classifiers commonly used in misinformation detection.

---
**UPDATE 27.06.2024**: Extending beyond the current arXiv preprint, the repository also includes:

- two additional victim models based on fine-tuned [Gemma](https://huggingface.co/docs/transformers/model_doc/gemma)
  model in the 2-billion and 7-billion version. These are configured to run
  through [QLoRA](https://proceedings.neurips.cc/paper_files/paper/2023/file/1feb87871436031bdc0f2beaa62a049b-Paper-Conference.pdf)
  on CUDA GPU.
- an additional task of detecting COVID-19 misinformation (C19) and a 'surprise' classifier based on adversarially-tuned
  RoBERTa. Both were prepared for the
  [Task 6: Robustness of Credibility Assessment with Adversarial Examples (InCrediblAE)](https://checkthat.gitlab.io/clef2024/task6/)
  shared task at
  [CheckThat! 2024](https://checkthat.gitlab.io/clef2024/) lab at CLEF-2024. The files for all the tasks and victims
  involved in
  this event are available at
  the [lab repository](https://gitlab.com/checkthat_lab/clef2024-checkthat-lab/-/tree/main/task6?ref_type=heads).
---

The research was done within the [ERINIA](https://www.upf.edu/web/erinia) project realised at the
[TALN lab](https://www.upf.edu/web/taln/) of [Universitat Pompeu Fabra](https://www.upf.edu).

## Installation

In order to use BODEGA, you will first need to prepare an environment with *python 3.10* and
install [HuggingFace transformers](https://huggingface.co/docs/transformers/index)
with [OpenAttack](https://openattack.readthedocs.io/):

```commandline
pip install "transformers==4.38.1"
pip install OpenAttack
```

You may wish to install the OpenAttack dependencies separately (including numpy, nltk and
pytorch). Note that some of the attack implementations will require additional packages -- for more information see
[OpenAttack website](https://openattack.readthedocs.io/en/latest/quickstart/installation.html).

Additionally, you will need [editdistance](https://github.com/roy-ht/editdistance),
[BERTScore](https://github.com/Tiiiger/bert_score) and [bleurt-pytorch](https://github.com/lucadiliello/bleurt-pytorch):

```commandline
pip install editdistance
pip install bert-score
pip install git+https://github.com/lucadiliello/bleurt-pytorch.git
```

To perform per-sentence similarity computation, you will
need [LAMBO](https://gitlab.clarin-pl.eu/syntactic-tools/lambo) segmenter:

```commandline
git clone https://gitlab.clarin-pl.eu/syntactic-tools/lambo.git
pip install ./lambo
```

Gemma victim models also require several packages for efficient training (if your machine lacks GPU, you will not be
able to install `bitsandbytes` and the Gemma victims will not work correctly) :

```commandline
pip install peft bitsandbytes accelerate
```

Now you can clone this repository and start working with BODEGA.

## Usage

Performing evaluation with BODEGA requires several steps. Note that you can skip points 1-3 by downloading the resources
from the [repository](https://gitlab.com/checkthat_lab/clef2024-checkthat-lab/-/tree/main/task6?ref_type=heads) of
the [InCrediblAE shared task](https://checkthat.gitlab.io/clef2024/task6/).

### 1. Obtaining the source corpora

All the basic corpora are available to download on open licences. Specifically, the tasks included rely on the
following datasets:

- News bias
  assessment: [Data for PAN at SemEval 2019 Task 4: Hyperpartisan News Detection](https://zenodo.org/record/5776081)
- Propaganda
  detection: [SemEval-2020 Task 11: Detection of Propaganda Techniques in News Articles](https://zenodo.org/record/3952415)
- Fact checking: [FEVER Dataset](https://fever.ai/dataset/fever.html) (training and shared task development (labelled)
  datasets, including pre-processed Wikipedia Pages)
- Rumour
  detection: [Augmented dataset of rumours and non-rumours for rumour detection](https://zenodo.org/record/3269768) (`aug-rnr-data_filtered`
  version)

### 2. Converting to train/attack datasets

The corpora need to be processed by the scripts in `conversion/convert_{TASK_ID}.py`. Each script takes an
uncompressed corpus (the first command-line argument), reads the data, converts to a uniform binary classification setup
and outputs three subsets in the desired location (the second command-line argument):

- `train.tsv` -- roughly 80% of the text, used for training a victim model,
- `attack.tsv` -- around 400 instances, used to test the attack,
- `dev.tsv` -- the remaining data, currently unused.

Each TSV file has three columns: the classification label (1: non-credible, 0: credible), the source URL (or identifier)
and content text (with newlines encoded as `\n`).

For example, if you've downloaded and uncompressed the propaganda corpus in `~/Downloads/BODEGA`, you can generate the
corresponding datasets in
the following way:

```commandline
python ./conversion/convert_PR2.py ~/Downloads/BODEGA/datasets ~/Downloads/BODEGA/
```

### 3. Training victim classifiers

Training victim classifiers is done using the `runs/train_victims.py` script. You need to provide the following
arguments:

- task ID (`HN`, `PR2`, `FC`, `RD` or `C19`),
- classifier type (`BiLSTM`, `BERT`, `GEMMA2B` or `GEMMA7B`),
- path to the folder containing the TSV files with training data,
- path to the output file with the trained model.

For example, training the BiLSTM classifier for the PR2 data can be done in the following way:

```commandline
python ./runs/train_victims.py PR2 BiLSTM ~/Downloads/BODEGA ~/Downloads/BODEGA/bilstm.pth
```

The code for training the surprise classifier is not available here.

### 4. Testing the attack performance

Running the attack and measuring its success is done through the `runs/attack.py` script. You need to provide the
following arguments:

- task ID (`HN`, `PR2`, `FC`, `RD` or `C19`),
- indication if the attack is targeted (`true` or `false`),
- attacker procedure (`PWWS`, `SCPN`, `TextFooler`, `DeepWordBug`, `GAN`, `Genetic`, `PSO`, `BERTattack` and
  `BAE` are currently supported)
- victim type (`BiLSTM`, `BERT`, `GEMMA2B`, `GEMMA7B` or `surprise`),
- path to the folder containing the TSV files with training data,
- victim model path.

For example, we can test the robustness of victim model generated in the previous step against BERattack in the
following way:

```commandline
python ./runs/attack.py PR2 true BERTattack BiLSTM ~/Downloads/BODEGA ~/Downloads/BODEGA/bilstm.pth 
```

At the end you should see the results as follows:

```commandline
Subset size: 50
Success score: 0.94
BERT score: 0.8319604
Levenshtein score: 0.9240004592303396
BODEGA score: 0.7248815366883989
Queries per example: 50.14
Total attack time: 5.048301696777344
Time per example: 0.10096603393554687
Total evaluation time: 11.681389093399048
```

### Extensions

The procedures described above allows you to replicate the results we include in the
[article preprint](https://arxiv.org/abs/2303.08032). You can however use the framework to test other solutions.
Two of the clear extensions are:

- Testing the robustness of your classifier. In order to do that, you need to implement the `OpenAttack.Classifier`
  interface. For an example see `VictimBiLSTM` in `victims/bilstm.py`
  or the [OpenAttack manual](https://openattack.readthedocs.io/en/latest/examples/example2.html). Now you can use your
  own classifier in `runs/attack.py`.
- Testing the performance of your own attack method. In order to do that, you need to implement
  the `OpenAttack.attackers.ClassificationAttacker` interface. See an example in
  the [OpenAttack manual](https://openattack.readthedocs.io/en/latest/examples/example3.html).

## Licence

BODEGA code is released under the [GNU GPL 3.0](https://www.gnu.org/licenses/gpl-3.0.html) licence.

## Funding

The [ERINIA](https://www.upf.edu/web/erinia) project has received funding from the European Union’s Horizon Europe
research and innovation programme under grant agreement No 101060930.

Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the
European Union. Neither the European Union nor the granting authority can be held responsible for them.

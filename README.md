# Hierarchical Causal Learning for Face Age Synthesis

This project is the official implementation of the paper _"**Hierarchical Causal Learning for Face Age Synthesis**"_.

We propose a novel **Hierarchical Causal Face Synthesis (HCFace)** framework that automatically discovers causal relationships among facial attributes and leverages them to guide age synthesis. 

## Key Features

- **Causal Graph Discovery Module**  
  Automatically models the causal relationships of facial attributes, discovers the causal structure among attributes, and constructs hierarchical causal graphs to guide subsequent age editing.

- **Non-linear Mapping Module**  
  Takes the discovered hierarchical causal graph as input and guides the model to modify attribute values along the causal paths, generating facial images that realistically reflect facial aging patterns across different age groups.

## Datasets

### Training Dataset

We train our model using the **MS1M** dataset, a large-scale face recognition dataset.

| Dataset                | Description                                                  | Download Link                                               |
| :--------------------- | :----------------------------------------------------------- | :---------------------------------------------------------- |
| **MS1M** (MS-Celeb-1M) | ~10M images, 100k identities. We use the cleaned `faces_emore` version. | [AI Studio](https://aistudio.baidu.com/datasetdetail/22814) |

### Testing Datasets

We evaluate our method on four public age synthesis / age progression benchmarks:

| Dataset    | Description                                                  | #Images | #Subjects | Download Link                                                |
| :--------- | :----------------------------------------------------------- | :-----: | :-------: | :----------------------------------------------------------- |
| **CACD**   | Cross-Age Celebrity Dataset                                  | 163,446 |   2,000   | [Link](https://bcsiriuschen.github.io/CARC/)                 |
| **FG-NET** | Face Aging Dataset                                           |  1,002  |    82     | [Link](https://yanweifu.github.io/FG_NET_data/)              |
| **MORPH2** | Longitudinal Face Dataset                                    | 55,134  |  13,618   | [Link](https://uncw.edu/oic/tech/morph.html)                 |
| **ECAF**   | dataset with a diverse age<br/>distribution, comprising 5,265 face images of 613 individuals,<br/>with an average age of 41.3 years. |    -    |     -     | [Link](https://drive.google.com/file/d/1t5O3qbkXi-nD6lQjSHMTeMS2Elp0qUi1/view?usp=share_link) |

## Setup

### Environment Requirements

- Python 3.8+
- PyTorch 1.8+
- CUDA 11.1+ (recommended)

### Installation

1. Clone this repository:
2. git clone https://github.com/SE-hash/HCFace.git
3. cd HCFace
4. pip install requirement.txt
5. follow main.py, Note that you should ensure that the number of compute cards you can use matches the number specified in `--nproc_per_node=x`


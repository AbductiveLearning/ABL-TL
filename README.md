

# Deciphering Raw Data in Neuro-Symbolic Learning with Provable Guarantees

This is the repository for holding the sample code of [Deciphering Raw Data in Neuro-Symbolic Learning with Provable Guarantees](https://arxiv.org/pdf/2308.10487.pdf)  in AAAI 2024.

## Getting Started

Our code relies on PyTorch, which will be automatically installed when you follow the instructions below.

```
conda create -n abl-tl python=3.8
conda activate abl-tl
pip install -r requirements.txt
```

## Running Experiments

- ABL-TL on ConjEq.

  ```
  python main.py --train_loss TL --kb ConjEq
  ```

- TL-Risk on Conjunction.

  ```
  python main.py --train_loss TL --kb Conjunction
  ```


## Citing this work

```
@inproceedings{tao2024deciphering,
  title={Deciphering Raw Data in Neuro-Symbolic Learning with Provable Guarantees},
  author={Tao, Lue and Huang, Yu-Xuan and Dai, Wang-Zhou and Jiang, Yuan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2024}
}
```
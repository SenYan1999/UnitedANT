# UnitedANT
This is the code for [UnitedANT]() model.

## About the Data
Download data from [here](), and unzip it at the root directory of this repo. The data are processed based on this [dataset](https://github.com/GeminiLn/EarningsCall_Dataset) which provides the acoustic and textual data, and we use OpenSMILE to extract the acoustic feature. The numeric data are from [CRSP](http://www.crsp.org/).

## Prepare Data
```python
python main.py --do_prepare
```

## Train Model
```python
python main.py --do_train --batch_size 4 --tau [tau]
```
where **[tau]** is the size of the window which is in [3, 7, 15, 30].

## Reference
[Cross-Modal BERT for Text-Audio Sentiment Analysis](https://github.com/thuiar/Cross-Modal-BERT), MM 2020

[Memory fusion network for multi-view sequential learning (MFN)](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/17341/16122), AAAI 2018

[Tensor fusion network for multimodal sentiment analysis (TFN)](https://www.aclweb.org/anthology/D17-1115.pdf), ACL 2017

[Multi-attention recurrent network for human communication comprehension (MARN)](https://arxiv.org/abs/1802.00923), AAAI 2018

[Multimodal Transformer for Unaligned Multimodal Language Sequences (MulT)](https://arxiv.org/pdf/1906.00295.pdf), ACL 2019, [Github](https://github.com/yaohungt/Multimodal-Transformer)

[Videobert: A joint model for video and language representation learning](https://arxiv.org/pdf/1904.01766.pdf), ICCV 2019

[Multimodal language analysis with recurrent multistage fusion (RMFN)](https://www.aclweb.org/anthology/D18-1014.pdf), ACL 2018

[Adapting BERT for Target-Oriented Multimodal Sentiment Classification](https://www.ijcai.org/Proceedings/2019/0751.pdf), IJCAI 2019

[Integrating Multimodal Information in Large Pretrained Transformers](https://arxiv.org/abs/1908.05787), ACL 2020

[M-bert: Injecting multimodal information in the bert structure](https://arxiv.org/abs/1908.05787), arXiv 2019

[Audio ALBERT: A Lite BERT for Self-supervised Learning of Audio Representation](https://arxiv.org/abs/2005.08575), arXiv 2020

[Multimodal big data affective analytics: A comprehensive survey using text,audio, visual and physiological signals](https://www.sciencedirect.com/science/article/pii/S1084804519303078), JNCA 2020

[Look, Listen, and Attend: Co-Attention Network for Self-Supervised Audio-Visual Representation Learning](https://arxiv.org/pdf/2008.05789.pdf), arXiv 2020

[Improved Fusion of Visual and Language Representations by Dense Symmetric Co-Attention for Visual Question Answering](https://arxiv.org/pdf/1804.00775.pdf), arXiv 2018

[Audio Sentiment Analysis by Heterogeneous Signal Features Learned from Utterance-Based Parallel Neural Network](http://ceur-ws.org/Vol-2328/3_2_paper_17.pdf), arXiv 2019

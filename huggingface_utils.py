"""
Author: Subhabrata Mukherjee (submukhe@microsoft.com)
Code for Uncertainty-aware Self-training (UST) for few-shot learning.
"""

from transformers import *

# HuggingFace Transformers has a unified API
# for 10 transformer architectures and 30 pretrained weights.
#          Model          | Tokenizer          | Pretrained model config
MODELS = [(TFAlbertModel, AlbertTokenizer, AlbertConfig),
          (TFBertModel, BertTokenizer, BertConfig),
          (TFElectraModel, ElectraTokenizer, ElectraConfig),
          (TFOpenAIGPTModel, OpenAIGPTTokenizer, OpenAIGPTConfig),
          (TFGPT2LMHeadModel, GPT2Tokenizer, GPT2Config),
          (TFCTRLModel, CTRLTokenizer, CTRLConfig),
          (TFTransfoXLModel,  TransfoXLTokenizer, TransfoXLConfig),
          (TFXLNetModel, XLNetTokenizer, XLNetConfig),
          (TFXLMModel, XLMTokenizer),
          (TFDistilBertModel, DistilBertTokenizer, DistilBertConfig),
          (TFRobertaModel, RobertaTokenizer, RobertaConfig),
          (TFXLMRobertaModel, XLMRobertaTokenizer, XLMRobertaConfig),
          (TFMobileBertForMaskedLM, MobileBertTokenizer, MobileBertConfig),
          (TFFunnelForMaskedLM, FunnelTokenizer, FunnelConfig)
          ]

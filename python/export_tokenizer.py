import sentencepiece.sentencepiece_model_pb2 as model_pb2
import struct
import os
import json
from contextlib import nullcontext
import torch

from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer


tokenizer=ChatGLMTokenizer(vocab_file='/home/hp/code/python/baby-llama2-chinese/chatglm_tokenizer/tokenizer.model')
modelInfo = {}
modelInfo["model_type"] = "chatglm"
fo = open("/home/hp/code/llm/tokenizer/workspace/token_model.bin", "wb")
def writeString(fo, s):
    bytes = s.encode()
    fo.write(struct.pack('i', len(bytes)))
    fo.write(bytes)

def writeKeyValue(fo, key, value):
    writeString(fo, key)
    writeString(fo, value)
# 1. vocab
if (tokenizer):
    if (hasattr(tokenizer, "tokenizer")):
        if modelInfo["model_type"] == "qwen":
            pass
        else:
            tokenizer = tokenizer.tokenizer
    if (hasattr(tokenizer, "sp_model")):
        piece_size = tokenizer.sp_model.piece_size()
        fo.write(struct.pack('i', piece_size))
        for i in range(piece_size):
            s = tokenizer.sp_model.id_to_piece(i).encode()
            fo.write(struct.pack('i', len(s)))
            for c in s:
                fo.write(struct.pack('i', c))
            fo.write(struct.pack('i', i))
            fo.write(struct.pack('f', float(tokenizer.sp_model.get_score(i))))
    else:
        if hasattr(tokenizer, "bpe_ranks"):
            merges = {("".join(bpe_tokens), token_index) for bpe_tokens, token_index in
                      sorted(tokenizer.bpe_ranks.items(), key=lambda kv: kv[1])}
        vocab = tokenizer.get_vocab()
        fo.write(struct.pack('i', len(vocab)))
        for v in vocab.keys():
            score = merges[v] if v in merges else 1.0
            # if (modelInfo["model_type"] == "moss"):
            #     s = [(ord(c) if c not in tokenizer.byte_decoder else tokenizer.byte_decoder[c]) for c in v]
            if (modelInfo["model_type"] == "qwen"):
                s = v
            else:
                s = v.encode()
            fo.write(struct.pack('i', len(s)))
            for c in s:
                fo.write(struct.pack('i', c))
            fo.write(struct.pack('i', vocab[v]))
            fo.write(struct.pack('f', score))
    if ("tokenizer_has_special_tokens" in modelInfo):
        fo.write(struct.pack('i', len(tokenizer.all_special_tokens)))
        for special_token in tokenizer.all_special_tokens:
            writeString(fo, special_token)
else:
    fo.write(struct.pack('i', 0))
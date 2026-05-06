import logging
import os

import jax
import numpy as np
import orbax.checkpoint as ocp
import sentencepiece
from transformers import AutoProcessor

import openpi.models.utils.fsq_tokenizer as fsq_tokenizer
import openpi.shared.download as download
import random

import sentencepiece

PATH_paligemma_tokenizer_model = "/home/rsluo/codes/openpi06/src/tokenizer_model/paligemma_tokenizer.model"
path = download.maybe_download(PATH_paligemma_tokenizer_model, gs={"token": "anon"})


with path.open("rb") as f:
    _tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
cleaned_text = "pick up the square paper box and put it on the plate"
cleaned_text = "insert sfp_port_0 on nic_card_mount_1"
# cleaned_text = "pick up a pack of tissues and put it on the plate"
# cleaned_text = "pick up the water and put it on the plate"
state_str = "24 152 134 84 135 78 105 176"
print_texts = [
    f"Task: {cleaned_text}, State: {state_str} ",
    f"Task: {cleaned_text}, State: {state_str};",
    f"Task: {cleaned_text}, State: {state_str};\n",
    f"Task: {cleaned_text}, State: {state_str};\nAction",
    f"Task: {cleaned_text}, State: {state_str};\nAction:",
    f"Task: {cleaned_text}, State: {state_str};\nAction: ",
    f"Task: {cleaned_text}, State: {state_str},\nAction: ",
    f"Task: {cleaned_text}, State: {state_str},;\nAction: ",
    # f"Task: {cleaned_text}, State: {state_str},",
    # f"Task: {cleaned_text}, State: {state_str},;",
    # f"Task: {cleaned_text}, State: {state_str}, ",
    # f"Task: {cleaned_text}, State: {state_str}, Indicator",
    # f"Task: {cleaned_text}, State: {state_str}, Indicator:",
    # f"Task: {cleaned_text}, State: {state_str}, Indicator: ",
    # f"Task: {cleaned_text}, State: {state_str}, Indicator: [",
    # f"Task: {cleaned_text}, State: {state_str}, Indicator: [INDICATOR",
    # f"Task: {cleaned_text}, State: {state_str}, Indicator: [INDICATOR]",
    # f"Task: {cleaned_text}, State: {state_str}, Indicator: [INDICATOR];",
    # f"Task: {cleaned_text}, State: {state_str}, Indicator: [INDICATOR];\n",
    # f"Task: {cleaned_text}, State: {state_str}, Indicator: [INDICATOR];\nAction",
    # f"Task: {cleaned_text}, State: {state_str}, Indicator: [INDICATOR];\nAction:",
    # f"Task: {cleaned_text}, State: {state_str}, Indicator: [INDICATOR];\nAction: ",
    # f"Task: {cleaned_text}, State: {state_str}, Indicator: [INDICATOR],Advantage: positive;\nAction: ",
    # f"Task: {cleaned_text}, State: {state_str}, Indicator: [INDICATOR],Advantage: negative;\nAction: ",
    # f"Task: {cleaned_text}, State: {state_str}, Indicator: [INDICATOR];Advantage: positive;\nAction: ",
    # f"Task: {cleaned_text}, State: {state_str}, Indicator: [INDICATOR];Advantage: negative;\nAction: ",
    # f"Task: {cleaned_text}, State: {state_str}; Indicator: [INDICATOR];Advantage: negative;\nAction: ",
    # f"Task: {cleaned_text}, State: {state_str}, Indicator: [ONE];\nAction: ",
    # f"Task: {cleaned_text}, State: {state_str}, Indicator: [ZERO];\nAction: ",
]
for text in print_texts:
    full_prompt = text
    tokens = _tokenizer.encode(full_prompt, add_bos=True)
    print(f"full___prompt = {full_prompt}")
    print(f"tokens({len(tokens)}) = {tokens}")
    # decode_prompt = _tokenizer.decode(tokens)
    # print(f"decode_prompt = {decode_prompt}")
    print("="*50)

# full_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR];
# Action: 
# tokens(57) = [一致, 235269, 77608, 235292, 892, 91858, 1380, 108, 4022, 235292, 235248]
# full_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR],Advantage: positive;
# Action: 
# tokens(61) = [一致, 235269, 77608, 235292, 892, 91858, 1308, 105985, 235292, 6222, 235289, 108, 4022, 235292, 235248]
# full_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR],Advantage: negative;
# Action: 
# tokens(61) = [一致, 235269, 77608, 235292, 892, 91858, 1308, 105985, 235292, 8322, 235289, 108, 4022, 235292, 235248]




# tokens(57) = [2, 7071, 235292, 4788, 908, 476, 3386, 576, 29703, 578, 2507, 665, 611, 573, 8811, 235269, 3040, 235292, 235248, 235284, 235310, 235248, 235274, 235308, 235284, 235248, 235274, 235304, 235310, 235248, 235321, 235310, 235248, 235274, 235304, 235308, 235248, 235324, 235321, 235248, 235274, 235276, 235308, 235248, 235274, 235324, 235318, 235269, 77608, 235292, 892, 91858, 1380, 108, 4022, 235292, 235248]
# full_prompt = Task: pick up a pack of tissues and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR],Advantage: positive;
# Action: 
# tokens(61) = [2, 7071, 235292, 4788, 908, 476, 3386, 576, 29703, 578, 2507, 665, 611, 573, 8811, 235269, 3040, 235292, 235248, 235284, 235310, 235248, 235274, 235308, 235284, 235248, 235274, 235304, 235310, 235248, 235321, 235310, 235248, 235274, 235304, 235308, 235248, 235324, 235321, 235248, 235274, 235276, 235308, 235248, 235274, 235324, 235318, 235269, 77608, 235292, 892, 91858, 1308, 105985, 235292, 6222, 235289, 108, 4022, 235292, 235248]
# full_prompt = Task: pick up a pack of tissues and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR],Advantage: negative;
# Action: 
# tokens(61) = [2, 7071, 235292, 4788, 908, 476, 3386, 576, 29703, 578, 2507, 665, 611, 573, 8811, 235269, 3040, 235292, 235248, 235284, 235310, 235248, 235274, 235308, 235284, 235248, 235274, 235304, 235310, 235248, 235321, 235310, 235248, 235274, 235304, 235308, 235248, 235324, 235321, 235248, 235274, 235276, 235308, 235248, 235274, 235324, 235318, 235269, 77608, 235292, 892, 91858, 1308, 105985, 235292, 8322, 235289, 108, 4022, 235292, 235248]



# tokens(55) = [2, 7071, 235292, 4788, 908, 573, 2003, 578, 2507, 665, 611, 573, 8811, 235269, 3040, 235292, 235248, 235284, 235310, 235248, 235274, 235308, 235284, 235248, 235274, 235304, 235310, 235248, 235321, 235310, 235248, 235274, 235304, 235308, 235248, 235324, 235321, 235248, 235274, 235276, 235308, 235248, 235274, 235324, 235318, 235269, 77608, 235292, 892, 91858, 1380, 108, 4022, 235292, 235248]
# full_prompt = Task: pick up the water and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR],Advantage: positive;
# Action: 
# tokens(59) = [2, 7071, 235292, 4788, 908, 573, 2003, 578, 2507, 665, 611, 573, 8811, 235269, 3040, 235292, 235248, 235284, 235310, 235248, 235274, 235308, 235284, 235248, 235274, 235304, 235310, 235248, 235321, 235310, 235248, 235274, 235304, 235308, 235248, 235324, 235321, 235248, 235274, 235276, 235308, 235248, 235274, 235324, 235318, 235269, 77608, 235292, 892, 91858, 1308, 105985, 235292, 6222, 235289, 108, 4022, 235292, 235248]
# full_prompt = Task: pick up the water and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR],Advantage: negative;
# Action: 
# tokens(59) = [2, 7071, 235292, 4788, 908, 573, 2003, 578, 2507, 665, 611, 573, 8811, 235269, 3040, 235292, 235248, 235284, 235310, 235248, 235274, 235308, 235284, 235248, 235274, 235304, 235310, 235248, 235321, 235310, 235248, 235274, 235304, 235308, 235248, 235324, 235321, 235248, 235274, 235276, 235308, 235248, 235274, 235324, 235318, 235269, 77608, 235292, 892, 91858, 1308, 105985, 235292, 8322, 235289, 108, 4022, 235292, 235248]
    


# full_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176 
# tokens(48) = [一致, 235248]
# full_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176;
# tokens(48) = [一致, 235289]
# full_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176;
# 
# tokens(49) = [一致, 235289, 108]
# full_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176;
# Action
# tokens(50) = [一致, 235289, 108, 4022]
# full_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176;
# Action:
# tokens(51) = [一致, 235289, 108, 4022, 235292]
# full_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176;
# Action: 
# tokens(52) = [一致, 235289, 108, 4022, 235292, 235248]
# full_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR];
# Action: 
# tokens(57) = [一致, 235269, 77608, 235292, 892, 91858, 1380, 108, 4022, 235292, 235248]
# full_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR],Advantage: positive;
# Action: 
# tokens(61) = [一致, 235269, 77608, 235292, 892, 91858, 1308, 105985, 235292, 6222, 235289, 108, 4022, 235292, 235248]
# full_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR],Advantage: negative;
# Action: 
# tokens(61) = [一致, 235269, 77608, 235292, 892, 91858, 1308, 105985, 235292, 8322, 235289, 108, 4022, 235292, 235248]






# 规则
# 1、数字和符号
# full_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176 
# tokens(48) = [一致, 235248]
# decode_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176 
# ==================================================
# full_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176;
# tokens(48) = [一致, 235289]
# decode_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176;
# ==================================================
# full_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176;
#
# tokens(49) = [一致, 235289, 108]
# decode_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176;
#
# ==================================================
# full_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176;
# Action
# tokens(50) = [一致, 235289, 108, 4022]
# decode_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176;
# Action
# ==================================================
# full_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176;
# Action:
# tokens(51) = [一致, 235289, 108, 4022, 235292]
# decode_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176;
# Action:
# ==================================================
# full_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176;
# Action: 
# tokens(52) = [一致, 235289, 108, 4022, 235292, 235248]
# decode_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176;
# Action: 
# ==================================================
# full_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176,
# Action: 
# tokens(52) = [一致, 235269, 108, 4022, 235292, 235248]
# decode_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176,
# Action: 
# ==================================================
# full_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176,;
# Action: 
# tokens(52) = [一致, 165191, 108, 4022, 235292, 235248]
# decode_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176,;
# Action: 
# ==================================================
# full_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176,
# tokens(48) = [一致, 235269]
# decode_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176,
# ==================================================
# full_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176,;
# tokens(48) = [一致, 165191]
# decode_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176,;
# ==================================================
# full_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, 
# tokens(49) = [一致, 235269, 235248]
# decode_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, 
# ==================================================
# full_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator
# tokens(49) = [一致, 235269, 77608]
# decode_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator
# ==================================================
# full_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator:
# tokens(50) = [一致, 235269, 77608, 235292]
# decode_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator:
# ==================================================
# full_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: 
# tokens(51) = [一致, 235269, 77608, 235292, 235248]
# decode_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: 
# ==================================================
# full_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [
# tokens(51) = [一致, 235269, 77608, 235292, 892]
# decode_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [
# ==================================================
# full_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR
# tokens(52) = [一致, 235269, 77608, 235292, 892, 91858]
# decode_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR
# ==================================================
# full_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR]
# tokens(53) = [一致, 235269, 77608, 235292, 892, 91858, 235307]
# decode_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR]
# ==================================================
# full_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR];
# tokens(53) = [一致, 235269, 77608, 235292, 892, 91858, 1380]
# decode_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR];
# ==================================================
# full_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR];
#
# tokens(54) = [一致, 235269, 77608, 235292, 892, 91858, 1380, 108]
# decode_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR];
#
# ==================================================
# full_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR];
# Action
# tokens(55) = [一致, 235269, 77608, 235292, 892, 91858, 1380, 108, 4022]
# decode_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR];
# Action
# ==================================================
# full_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR];
# Action:
# tokens(56) = [一致, 235269, 77608, 235292, 892, 91858, 1380, 108, 4022, 235292]
# decode_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR];
# Action:
# ==================================================
# full_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR];
# Action: 
# tokens(57) = [一致, 235269, 77608, 235292, 892, 91858, 1380, 108, 4022, 235292, 235248]
# decode_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR];
# Action: 
# ==================================================
# full_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR],Advantage: positive;
# Action: 
# tokens(61) = [一致, 235269, 77608, 235292, 892, 91858, 1308, 105985, 235292, 6222, 235289, 108, 4022, 235292, 235248]
# decode_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR],Advantage: positive;
# Action: 
# ==================================================
# full_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR],Advantage: negative;
# Action: 
# tokens(61) = [一致, 235269, 77608, 235292, 892, 91858, 1308, 105985, 235292, 8322, 235289, 108, 4022, 235292, 235248]
# decode_prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR],Advantage: negative;
# Action: 
# ==================================================
    

# full___prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176 
# tokens(48) = [一致, 235248]
# ==================================================
# full___prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176;
# tokens(48) = [一致, 235289]
# ==================================================
# full___prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176;

# tokens(49) = [一致, 235289, 108]
# ==================================================
# full___prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176;
# Action
# tokens(50) = [一致, 235289, 108, 4022]
# ==================================================
# full___prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176;
# Action:
# tokens(51) = [一致, 235289, 108, 4022, 235292]
# ==================================================
# full___prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176;
# Action: 
# tokens(52) = [一致, 235289, 108, 4022, 235292, 235248]
# ==================================================
# full___prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176,
# Action: 
# tokens(52) = [一致, 235269, 108, 4022, 235292, 235248]
# ==================================================
# full___prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176,;
# Action: 
# tokens(52) = [一致, 165191, 108, 4022, 235292, 235248]
# ==================================================
# full___prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176,
# tokens(48) = [一致, 235269]
# ==================================================
# full___prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176,;
# tokens(48) = [一致, 165191]
# ==================================================
# full___prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, 
# tokens(49) = [一致, 235269, 235248]
# ==================================================
# full___prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator
# tokens(49) = [一致, 235269, 77608]
# ==================================================
# full___prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator:
# tokens(50) = [一致, 235269, 77608, 235292]
# ==================================================
# full___prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: 
# tokens(51) = [一致, 235269, 77608, 235292, 235248]
# ==================================================
# full___prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [
# tokens(51) = [一致, 235269, 77608, 235292, 892]
# ==================================================
# full___prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR
# tokens(52) = [一致, 235269, 77608, 235292, 892, 91858]
# ==================================================
# full___prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR]
# tokens(53) = [一致, 235269, 77608, 235292, 892, 91858, 235307]
# ==================================================
# full___prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR];
# tokens(53) = [一致, 235269, 77608, 235292, 892, 91858, 1380]
# ==================================================
# full___prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR];

# tokens(54) = [一致, 235269, 77608, 235292, 892, 91858, 1380, 108]
# ==================================================
# full___prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR];
# Action
# tokens(55) = [一致, 235269, 77608, 235292, 892, 91858, 1380, 108, 4022]
# ==================================================
# full___prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR];
# Action:
# tokens(56) = [一致, 235269, 77608, 235292, 892, 91858, 1380, 108, 4022, 235292]
# ==================================================
# full___prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR];
# Action: 
# tokens(57) = [一致, 235269, 77608, 235292, 892, 91858, 1380, 108, 4022, 235292, 235248]
# ==================================================
# full___prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR],Advantage: positive;
# Action: 
# tokens(61) = [一致, 235269, 77608, 235292, 892, 91858, 1308, 105985, 235292, 6222, 235289, 108, 4022, 235292, 235248]
# ==================================================
# full___prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR],Advantage: negative;
# Action: 
# tokens(61) = [一致, 235269, 77608, 235292, 892, 91858, 1308, 105985, 235292, 8322, 235289, 108, 4022, 235292, 235248]
# ==================================================
# full___prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR];Advantage: positive;
# Action: 
# tokens(61) = [一致, 235269, 77608, 235292, 892, 91858, 1380, 105985, 235292, 6222, 235289, 108, 4022, 235292, 235248]
# ==================================================
# full___prompt = Task: pick up the square paper box and put it on the plate, State: 24 152 134 84 135 78 105 176, Indicator: [INDICATOR];Advantage: negative;
# Action: 
# tokens(61) = [一致, 235269, 77608, 235292, 892, 91858, 1380, 105985, 235292, 8322, 235289, 108, 4022, 235292, 235248]
# ==================================================



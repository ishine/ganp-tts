import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ElectraModel, ElectraConfig

from transformers import AutoConfig
from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths
from text.symbols import symbols

class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        prefileconfig = "config.json" 
        prefile = "pytorch_model.bin"
        config = AutoConfig.from_pretrained(
           prefileconfig,
        )
        self.encoder = ElectraModel.from_pretrained(
           prefile,
        #from_tf=bool(".ckpt" in args.model_name_or_path),
           config=config,
        )
        self.decoder_new = Decoder(model_config)

        self.mel_linear_dec = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            model_config["transformer"]["decoder_hidden"] - preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        
#        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()
        self.silienceid = [symbols.index("@sp"),symbols.index("@spn"),symbols.index("@sil")]

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
           
        src_masks_new = get_mask_from_lengths(src_lens, max_src_len)
        #max_src_len = p_targets.shape[-1]
        #print(max_src_len)
        max_src_len = max_src_len - 2 
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        #print("==================================texts.shape:",texts.shape)
        #print("src_mask.shape:",src_masks.shape)
        masks_new = 1 - src_masks_new.int()
        #print(masks_new) 
        seg_id = masks_new
        
        #print("src_mask_new",src_masks_new)
        #print(texts)
       # print(src_masks)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        #output = self.encoder(texts, src_masks)
        outputs = self.encoder(input_ids=texts, attention_mask=masks_new,token_type_ids=seg_id,output_hidden_states=True)
        output  = outputs[0]
     #   print(output.shape)
        outputshape1 = output.shape[-2]
        #print(outputshape1)
        _,output,_ = torch.split(output,[1,max_src_len,1],dim=-2)
      #  print(output.shape)
        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            texts,
            p_control,
            e_control,
            d_control,
        )

        output_encoder = self.mel_linear_dec(output)

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        output_new = torch.cat([output,output_encoder],-1)
        output_new, mel_masks = self.decoder_new(output_new,mel_masks)
        output_new = self.mel_linear(output_new)

        postnet_output = self.postnet(output_new) + output_new

        #postnet_output = self.postnet(output) + output


        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )

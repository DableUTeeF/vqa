from transformers import ViTImageProcessor, LlamaForCausalLM, AutoTokenizer, VisionEncoderDecoderModel
import torch
from peft import PeftModel
from PIL import Image
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.generation.stopping_criteria import validate_stopping_criteria, StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList
import os
from transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder import shift_tokens_right
from matplotlib import pyplot as plt
import json
from dataset.vqa_data import *
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    ViTImageProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
import torch
import os
import json
import argparse


def forward(
        self,
        pixel_values=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
):
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

    kwargs_decoder = {
        argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
    }
    if hasattr(self.decoder, 'transformer'):
        embeded_instruction = self.decoder.transformer.wte(instruction_tokens)
    elif hasattr(self.decoder, 'roberta'):
        embeded_instruction = self.decoder.roberta.embeddings(instruction_tokens)
    if encoder_outputs is None:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        encoder_outputs = self.encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs_encoder,
        )
    elif isinstance(encoder_outputs, tuple):
        encoder_outputs = BaseModelOutput(*encoder_outputs)

    encoder_hidden_states = encoder_outputs[0]
    encoder_hidden_states = torch.cat((encoder_hidden_states, embeded_instruction), 1)

    # optionally project encoder_hidden_states
    if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
    ):
        encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

    # else:
    encoder_attention_mask = None

    # Decode
    decoder_outputs = self.decoder(
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        encoder_hidden_states=encoder_hidden_states,  # n, 50, 768
        encoder_attention_mask=encoder_attention_mask,
        inputs_embeds=decoder_inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        use_cache=use_cache,
        past_key_values=past_key_values,
        return_dict=return_dict,
        **kwargs_decoder,
    )

    if not return_dict:
        return decoder_outputs + encoder_outputs

    return Seq2SeqLMOutput(
        logits=decoder_outputs.logits,
        past_key_values=decoder_outputs.past_key_values,
        decoder_hidden_states=decoder_outputs.hidden_states,
        decoder_attentions=decoder_outputs.attentions,
        cross_attentions=decoder_outputs.cross_attentions,
        encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        encoder_hidden_states=encoder_outputs.hidden_states,
        encoder_attentions=encoder_outputs.attentions,
    )

def preprocess(image, image_processor):
    image = Image.open(image).convert('RGB')

    model_inputs = image_processor(images=image, return_tensors='pt')

    return model_inputs


def fwd(model, model_inputs, generate_kwargs=None):
    if generate_kwargs is None:
        generate_kwargs = {'max_new_tokens': 50}
    inputs = model_inputs.pop(model.main_input_name)
    model_outputs = model.generate(inputs, **model_inputs, **generate_kwargs)
    return model_outputs


def postprocess(model_outputs):
    records = []
    for output_ids in model_outputs:
        record = {
            "generated_text": tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
            )
        }
        records.append(record)
    return records


def generate(inputs):
    model_inputs = image_processor(images=inputs, return_tensors='pt').to('cuda')
    model_outputs = fwd(model, model_inputs)
    outputs = postprocess(model_outputs)
    return outputs


parser = argparse.ArgumentParser()
parser.add_argument('cp', type=str)
parser.add_argument('output', type=str)
parser.add_argument('question', type=str)
args = parser.parse_args()

VisionEncoderDecoderModel.forward = forward
image_src = '/project/lt200060-capgen/coco/images'

device = 'cuda'
# cp = '/project/lt200203-aimedi/palm/vqas/workdir/gpt2-cross2_1.0e-05_8/checkpoint-520000'
vit_model = "/project/lt200060-capgen/palm/huggingface/vit-base-patch16-224-in21k"
model = VisionEncoderDecoderModel.from_pretrained(args.cp, device_map="cuda")
image_processor = ViTImageProcessor.from_pretrained(vit_model)
tokenizer = AutoTokenizer.from_pretrained(args.cp, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

model.config.eos_token_id = tokenizer.eos_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# question_json = '/project/lt200203-aimedi/vqav2/annotations/v2_OpenEnded_mscoco_test-dev2015_questions.json'
test_std_questions = json.load(open(args.question))
results_std = []
with torch.no_grad():
    for data in test_std_questions['questions']:
        image_id = data['image_id']
        question = data['question']
        question_id = data['question_id']
        image = Image.open(os.path.join(image_src, 'test2015', f'COCO_test2015_{image_id:012d}.jpg')).convert('RGB')
        instruction_tokens = tokenizer(
            [question], 
            padding=True,
            return_tensors="pt",
            truncation=True,
        ).input_ids.to('cuda')
        text = generate(image)
        result = {
            "question_id": int(question_id),
            "answer": text[0]['generated_text'],
        }
        results_std.append(result)

# output = 'coco-test-dev-2015_sbatch_cross2-520000.json'
json.dump(
    results_std,
    open(args.output, 'w')
)

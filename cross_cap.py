import os
from typing import Optional
import torch
import argparse
from dataset.vqa_data import *

from transformers import (
    AutoTokenizer,
    ViTImageProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from PIL import Image
import numpy as np
import evaluate
import nltk
from transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder import shift_tokens_right, VisionEncoderDecoderModel
from transformers.trainer_callback import ProgressCallback, PrinterCallback
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from torch.nn import CrossEntropyLoss


def forward(
        self,
        instruction_tokens,
        pixel_values: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
):
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

    kwargs_decoder = {
        argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
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

    if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
        decoder_input_ids = shift_tokens_right(
            labels, self.config.pad_token_id, self.config.decoder_start_token_id
        )

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

    # Compute loss independent from decoder (as some shift the logits inside them)
    loss = None
    if labels is not None:
        logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.reshape(-1))

    if not return_dict:
        if loss is not None:
            return (loss,) + decoder_outputs + encoder_outputs
        else:
            return decoder_outputs + encoder_outputs

    return Seq2SeqLMOutput(
        loss=loss,
        logits=decoder_outputs.logits,
        past_key_values=decoder_outputs.past_key_values,
        decoder_hidden_states=decoder_outputs.hidden_states,
        decoder_attentions=decoder_outputs.attentions,
        cross_attentions=decoder_outputs.cross_attentions,
        encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        encoder_hidden_states=encoder_outputs.hidden_states,
        encoder_attentions=encoder_outputs.attentions,
    )


def on_log(self, args, state, control, logs=None, **kwargs):
    if state.is_local_process_zero and self.training_bar is not None:
        _ = logs.pop("total_flos", None)


def tokenization_fn(captions, max_target_length=120):
    """Run tokenization on captions."""
    labels = tokenizer(captions,
                       padding="max_length",
                       max_length=max_target_length,
                       return_tensors="pt",
                       truncation=True).input_ids

    return labels


def feature_extraction_fn(image_paths):
    # images = [ ]
    # print(image_paths, flush=True)
    # for image_file in image_paths:
    #     print(image_file, flush=True)
    #     images.append(Image.open(image_file).convert('RGB'))

    encoder_inputs = feature_extractor(images=image_paths, return_tensors="pt")

    return encoder_inputs.pixel_values


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels


def collate_fn(batch):
    model_inputs = {'labels': [], 'pixel_values': [], 'instruction_tokens': []}
    for obj in batch:
        model_inputs['labels'].append(obj[2])
        model_inputs['instruction_tokens'].append(obj[1])
        model_inputs['pixel_values'].append(obj[0])
    model_inputs['labels'] = tokenization_fn(model_inputs['labels'], args.max_target_length)
    model_inputs['instruction_tokens'] = tokenizer(
        model_inputs['instruction_tokens'],
        padding=True,
        return_tensors="pt",
        truncation=True
    ).input_ids
    model_inputs['pixel_values'] = feature_extraction_fn(model_inputs['pixel_values'])
    return model_inputs


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = preds.argmax(axis=-1)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds,
                                                     decoded_labels)
    rouge_result = rouge.compute(predictions=decoded_preds,
                                 references=decoded_labels,
                                 use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in rouge_result.items()}
    # bleu_result = bleu.compute(predictions=decoded_preds,
    #                            references=decoded_labels)
    # result.update({k: round(v * 100, 4) for k, v in bleu_result.items()})
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result


def preprocess_logits_for_metrics(logits, labels):
    # print(logits[0].size(), labels.size())
    return logits[0].argmax(axis=-1), labels


if __name__ == '__main__':
    ProgressCallback.on_log = on_log
    PrinterCallback.on_log = on_log
    VisionEncoderDecoderModel.forward = forward
    parser = argparse.ArgumentParser()
    parser.add_argument('expname', type=str)
    parser.add_argument('dataset', type=str)
    parser.add_argument('--max_target_length', type=int, default=120)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--gradient_steps', type=int, default=4)
    parser.add_argument('--text_decode_model', type=str, default='gpt2')
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--encoder_freeze', action='store_true')
    args = parser.parse_args()
    expname = args.expname + f'_{args.lr:.1e}_{args.bs}_{args.dataset}'
    print(expname, flush=True)
    target_modules = ['enc_to_dec_proj', "q_proj", "v_proj",]
    if os.path.exists("/project/lt200060-capgen/palm/"):
        vit_model = "/project/lt200060-capgen/palm/huggingface/vit-base-patch16-224-in21k"
        text_decode_model = os.path.join("/project/lt200060-capgen/palm/huggingface/", args.text_decode_model)
        if args.text_decode_model == 'tinygpt':
            text_decode_model = '/project/lt200060-capgen/palm/capocr/workdir/tinygpt_distilled_256_8_0.5_32_8_distil_prerained_mse/train/checkpoint-905000'
        rouge_path = '/home/nhongcha/hf-caption/rouge/'
        output_dir = os.path.join('workdir/', expname)
        disable_tqdm = True
        worker = 16
        instances2017 = [
            '/project/lt200060-capgen/coco/annotations/instances_train2017.json',
            '/project/lt200060-capgen/coco/annotations/instances_val2017.json',
        ]
        qa_anns2014 = [
            '/project/lt200203-aimedi/vqav2/annotations/v2_mscoco_train2014_annotations.json',
            '/project/lt200203-aimedi/vqav2/annotations/v2_mscoco_val2014_annotations.json',
        ]
        qa_questions2014 = [
            '/project/lt200203-aimedi/vqav2/annotations/v2_OpenEnded_mscoco_train2014_questions.json',
            '/project/lt200203-aimedi/vqav2/annotations/v2_OpenEnded_mscoco_val2014_questions.json',
        ]
        vqav2_src = '/project/lt200060-capgen/coco/images'
    elif os.path.exists("/data"):
        vit_model = "google/vit-base-patch16-224-in21k"
        text_decode_model = args.text_decode_model
        rouge_path = 'rouge'
        output_dir = os.path.join('workdir/', expname)
        disable_tqdm = True
        worker = 16
        instances2017 = [
            '/home/palm/data/coco/annotations/instances_train2017.json',
            '/home/palm/data/coco/annotations/instances_val2017.json',
        ]
        qa_anns2014 = [
            '/home/palm/data/vqav2/v2_mscoco_train2014_annotations.json',
            '/home/palm/data/vqav2/v2_mscoco_val2014_annotations.json',
        ]
        qa_questions2014 = [
            '/home/palm/data/vqav2/v2_OpenEnded_mscoco_train2014_questions.json',
            '/home/palm/data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json',
        ]
        vqav2_src = '/home/palm/data/coco/images'
        gqa_train = '/data/gqa/annotations/train.csv'
        gqa_val = '/data/gqa/annotations/val.csv'
        gqa_src = '/data/gqa/images'

    tokenizer = AutoTokenizer.from_pretrained(text_decode_model, trust_remote_code=True)
    if args.dataset == 'vqa' or args.dataset == 'vqav2':
        datasets = VQAv2Datasets(
            instances2017,
            qa_anns2014,
            qa_questions2014,
            vqav2_src,
        )
        train_set = datasets.train2014()
        valid_set = datasets.val2014()
    elif args.dataset == 'gqa':
        train_set = GQADataset(gqa_src, gqa_train)
        valid_set = GQADataset(gqa_src, gqa_val)

    print(len(train_set), flush=True)
    print(len(valid_set), flush=True)
    logdir = os.path.join(args.logdir, expname)
    rouge = evaluate.load(rouge_path)
    if 'gpt' in args.text_decode_model.lower():
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    base_model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(vit_model, text_decode_model)
    base_model.config.eos_token_id = tokenizer.eos_token_id
    base_model.config.decoder_start_token_id = tokenizer.bos_token_id
    base_model.config.pad_token_id = tokenizer.pad_token_id
    feature_extractor = ViTImageProcessor.from_pretrained(vit_model)
    if args.encoder_freeze:
        for param in base_model.encoder.parameters():
            param.requires_grad = False

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.gradient_steps,
        per_device_eval_batch_size=1,
        learning_rate=args.lr,
        logging_steps=100,
        # max_steps=conf.max_steps,
        num_train_epochs=12,
        # report_to=conf.log_with,
        save_steps=5000,
        save_total_limit=1,
        logging_dir=logdir,
        warmup_steps=1000,
        warmup_ratio=1e-3,
        lr_scheduler_type='cosine',
        optim='adamw_torch',
        weight_decay=0.05,
        bf16=True,
        remove_unused_columns=True,
        gradient_checkpointing=True,
        run_name=expname,
        ddp_find_unused_parameters=False,
        disable_tqdm=disable_tqdm,
        evaluation_strategy="epoch",
        dataloader_num_workers=worker,
        # eval_steps=50000,
        save_safetensors=False,
        generation_max_length=args.max_target_length,

    )

    trainer = Seq2SeqTrainer(
        model=base_model,
        train_dataset=train_set,
        eval_dataset=valid_set,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collate_fn,
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(output_dir)

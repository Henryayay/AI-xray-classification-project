"""Vision-language model inference by label log-likelihood, plus LoRA fine-tuning.

Instead of generating free text and string-matching it (which yields no score,
silently maps unparseable answers to errors, and is sensitive to substring
order such as "COVID-19" inside "Non-COVID"), every class name is scored as
the assistant's continuation:

    s_c = sum_t log p(label_c[t] | prompt, image(s), label_c[<t])

and the positive-class logit is ``s_1 - s_0``. The continuation includes the
end-of-turn token so that a label which is a prefix of another is not favoured.
The scored text is exactly what LoRA training teaches the model to emit.
"""

import copy
import re

import numpy as np
import torch
from PIL import Image

from . import prompts

END_OF_TURN = "<end_of_turn>"


def load_vlm(model_id, device="cuda", dtype=torch.bfloat16, adapter_dir=None,
             attn_implementation="eager", revision=None):
    """Load a Gemma-3-family VLM and processor with Hugging Face Transformers.

    The same loader is used for zero-shot, ICL and fine-tuned scoring so that
    the inference path is identical across conditions; LoRA adapters trained
    with Unsloth are standard PEFT adapters and are attached here.
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation=attn_implementation, revision=revision
    ).to(device)
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()
    model.eval()
    return model, processor


def blank_like(img: Image.Image, value=128) -> Image.Image:
    """Content-free query image for contextual calibration (uniform mid-grey)."""
    return Image.new(img.mode if img.mode in ("L", "RGB") else "RGB", img.size, color=value)


def _rgb(img):
    return img.convert("RGB") if img.mode != "RGB" else img


class LabelScorer:
    def __init__(self, model, processor, class_names, use_prefix_cache=True):
        self.model = model
        self.processor = processor
        self.tok = processor.tokenizer
        self.class_names = list(class_names)
        self.use_prefix_cache = use_prefix_cache
        self.cont_ids = [self.tok(c + END_OF_TURN, add_special_tokens=False,
                                  return_tensors="pt").input_ids[0] for c in self.class_names]
        self.device = next(model.parameters()).device

    def _prompt_text(self, messages):
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True,
                                                  tokenize=False)
        return text

    def _encode(self, texts, images):
        return self.processor(text=texts, images=images, return_tensors="pt", padding=True,
                              add_special_tokens=False).to(self.device)

    @torch.no_grad()
    def score(self, messages, images):
        """Return per-class summed log-probabilities for one query."""
        images = [_rgb(im) for im in images]
        prompt = self._prompt_text(messages)
        if self.use_prefix_cache:
            return self._score_cached(prompt, images)
        return self._score_naive(prompt, images)

    def _score_naive(self, prompt, images):
        old_side = self.tok.padding_side
        self.tok.padding_side = "right"
        try:
            texts = [prompt + c + END_OF_TURN for c in self.class_names]
            enc = self._encode(texts, [images] * len(texts))
            logits = self.model(**enc).logits.float()
        finally:
            self.tok.padding_side = old_side
        logp = torch.log_softmax(logits, dim=-1)
        out = []
        lengths = enc["attention_mask"].sum(1)
        for r, ids in enumerate(self.cont_ids):
            L, end = len(ids), int(lengths[r])
            target = enc["input_ids"][r, end - L:end]
            if not torch.equal(target.cpu(), ids):
                raise RuntimeError("label tokenisation is not stable at the prompt boundary")
            pos = torch.arange(end - L - 1, end - 1, device=logp.device)
            out.append(float(logp[r, pos, target].sum()))
        return np.array(out)

    def _score_cached(self, prompt, images):
        enc = self._encode([prompt], [images])
        P = enc["input_ids"].shape[1]
        out = self.model(**enc, use_cache=True)
        first = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
        scores = []
        for ids in self.cont_ids:
            ids_d = ids.to(self.device)[None]
            s = float(first[ids_d[0, 0]])
            if ids_d.shape[1] > 1:
                cache = copy.deepcopy(out.past_key_values)
                step = ids_d[:, :-1]
                attn = torch.ones(1, P + step.shape[1], dtype=enc["attention_mask"].dtype,
                                  device=self.device)
                o2 = self.model(input_ids=step, past_key_values=cache, attention_mask=attn,
                                use_cache=True,
                                cache_position=torch.arange(P, P + step.shape[1],
                                                            device=self.device))
                lp = torch.log_softmax(o2.logits[0].float(), dim=-1)
                s += float(lp[torch.arange(step.shape[1]), ids_d[0, 1:]].sum())
            scores.append(s)
        return np.array(scores)

    @torch.no_grad()
    def generate(self, messages, images, max_new_tokens=15):
        """Greedy decoding, kept only to compare with the original string-parsing protocol."""
        enc = self._encode([self._prompt_text(messages)], [[_rgb(im) for im in images]])
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def parse_label(text, class_names):
    """Map free text to a class index, or -1. Longer/negated names are matched first."""
    t = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    norm = [(i, re.sub(r"[^a-z0-9]+", " ", c.lower()).strip()) for i, c in enumerate(class_names)]
    for i, c in sorted(norm, key=lambda x: -len(x[1])):
        if re.search(rf"\b{re.escape(c)}\b", t):
            return i
    return -1


def zero_shot_messages(task):
    return prompts.zero_shot_messages(task)


def train_lora(task, train_split, val_split, cfg, seed, output_dir, train_indices=None, log=print):
    """LoRA fine-tuning with Unsloth/TRL, mirroring the original recipe.

    Differences from the original notebooks, all deliberate: the prompt is the
    same one used at evaluation (``prompts.instruction``), checkpoints are kept
    locally (no Hub push), the seed varies per run, and 16-bit LoRA is the
    default so that scoring with a bf16 base matches the trained weights.
    Returns the adapter directory and the trainer log history.
    """
    from unsloth import FastVisionModel, UnslothVisionDataCollator  # noqa: I001
    from datasets import Dataset
    from transformers import EarlyStoppingCallback
    from trl import SFTConfig, SFTTrainer

    names = task.class_names
    idx = range(len(train_split)) if train_indices is None else train_indices
    conv_train = [prompts.training_conversation(task, train_split[int(i)]["image"],
                                                names[int(train_split[int(i)]["label"])])
                  for i in idx]
    conv_val = [prompts.training_conversation(task, ex["image"], names[int(ex["label"])])
                for ex in val_split]

    model, processor = FastVisionModel.from_pretrained(
        cfg["model_id"], revision=cfg.get("model_revision"),
        load_in_4bit=cfg.get("load_in_4bit", False),
        use_gradient_checkpointing="unsloth")
    model = FastVisionModel.get_peft_model(
        model, finetune_vision_layers=cfg.get("finetune_vision_layers", True),
        finetune_language_layers=True, finetune_attention_modules=True,
        finetune_mlp_modules=True, r=cfg.get("lora_r", 32),
        lora_alpha=cfg.get("lora_alpha", 32), lora_dropout=cfg.get("lora_dropout", 0.05),
        bias="none", random_state=seed, use_rslora=False, target_modules="all-linear")
    collator = UnslothVisionDataCollator(
        model=model, processor=processor, train_on_responses_only=True,
        instruction_part="<start_of_turn>user\n", response_part="<start_of_turn>model\n")
    FastVisionModel.for_training(model)
    trainer = SFTTrainer(
        model=model, train_dataset=Dataset.from_list(conv_train),
        eval_dataset=Dataset.from_list(conv_val), data_collator=collator,
        processing_class=processor.tokenizer,
        args=SFTConfig(
            per_device_train_batch_size=cfg.get("batch_size", 8),
            gradient_accumulation_steps=cfg.get("grad_accum", 2),
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_grad_norm=0.3, warmup_ratio=0.1, num_train_epochs=cfg.get("epochs", 10),
            learning_rate=cfg.get("lr", 2e-4), logging_steps=5, eval_strategy="epoch",
            save_strategy="epoch", load_best_model_at_end=True,
            metric_for_best_model="eval_loss", greater_is_better=False,
            output_dir=str(output_dir / "checkpoints"), save_total_limit=2, push_to_hub=False,
            report_to="none", optim="adamw_torch_fused", weight_decay=0.01,
            lr_scheduler_type="cosine", seed=seed, data_seed=seed, bf16=True,
            remove_unused_columns=False, dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True}, max_seq_length=2048),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.get("patience", 5))],
    )
    trainer.train()
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    history = trainer.state.log_history
    best = {"best_checkpoint": trainer.state.best_model_checkpoint,
            "best_eval_loss": trainer.state.best_metric}
    del trainer, model
    torch.cuda.empty_cache()
    return adapter_dir, history, best

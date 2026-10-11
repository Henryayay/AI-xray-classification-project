"""Tiny randomly-initialised Gemma-3 VLM with the real Gemma-3 processor (CPU tests only)."""

import torch

PROCESSOR_ID = "unsloth/gemma-3-4b-it"
PROCESSOR_REV = "bf46152c47f5dd20b896357cb51abc4c03b8ee8c"


def tiny_gemma3():
    from transformers import AutoProcessor, Gemma3Config, Gemma3ForConditionalGeneration

    proc = AutoProcessor.from_pretrained(PROCESSOR_ID, revision=PROCESSOR_REV)
    proc.image_processor.size = {"height": 56, "width": 56}
    tok = proc.tokenizer
    text_cfg = dict(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                    num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                    head_dim=16, sliding_window=64, max_position_embeddings=8192)
    vision_cfg = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=2, image_size=56, patch_size=14)
    mm_tokens = 4
    proc.image_seq_length = mm_tokens
    proc.full_image_sequence = (f"\n\n{proc.boi_token}" + proc.image_token * mm_tokens
                                + f"{tok.eoi_token}\n\n")
    cfg = Gemma3Config(text_config=text_cfg, vision_config=vision_cfg,
                       mm_tokens_per_image=mm_tokens,
                       image_token_index=tok.convert_tokens_to_ids(proc.image_token),
                       boi_token_index=tok.convert_tokens_to_ids(proc.boi_token),
                       eoi_token_index=tok.convert_tokens_to_ids(tok.eoi_token))
    torch.manual_seed(0)
    model = Gemma3ForConditionalGeneration(cfg).eval()
    return model, proc

"""검증한 백본 revision에 LoRA와 2출력 분류층을 추가한다."""
from __future__ import annotations

from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from torch import nn
from transformers import AutoModel


def backbone_layout(backbone_config, config):
    expected = {"dinov2": (14, 0), "dinov3_vit": (16, 4)}
    family = config.get("model_type", "dinov2")
    if family not in expected:
        raise ValueError(f"지원하지 않는 백본: {family}")
    patch, registers = expected[family]
    if (backbone_config.model_type != family or backbone_config.patch_size != patch
            or backbone_config.hidden_size != 768 or backbone_config.num_hidden_layers != 12
            or getattr(backbone_config, "num_register_tokens", 0) != registers):
        raise ValueError(f"예상한 {family} base 구조가 아님")
    if config["input_size"] <= 0 or config["input_size"] % patch:
        raise ValueError(f"입력 크기는 패치 크기 {patch}의 양의 배수여야 함")
    return patch, registers


POOLING_BLOCKS = {"cls_mean_max": 3, "cls_mean": 2}


def pooled_features(hidden, registers=0, pooling="cls_mean_max"):
    if pooling not in POOLING_BLOCKS:
        raise ValueError(f"지원하지 않는 특징 조합: {pooling}")
    patches = hidden[:, 1 + registers:]
    if patches.shape[1] == 0:
        raise ValueError("패치 토큰이 없음")
    features = [hidden[:, 0], patches.mean(1)]
    if pooling == "cls_mean_max":
        features.append(patches.amax(1))
    return torch.cat(features, dim=1)


class DamageDINO(nn.Module):
    def __init__(self, config, backbone_path=None, checkpointing=False):
        super().__init__()
        source = str(backbone_path or config["model_id"])
        kwargs = {"attn_implementation": "sdpa"}
        if backbone_path:
            if Path(source).name != config["revision"]:
                raise ValueError("캐시 snapshot과 revision 불일치")
            kwargs["local_files_only"] = True
        else:
            kwargs["revision"] = config["revision"]
        backbone = AutoModel.from_pretrained(source, **kwargs)
        self.patch_size, self.registers = backbone_layout(backbone.config, config)
        self.input_size = config["input_size"]
        self.pooling = config.get("pooling", "cls_mean_max")
        if self.pooling not in POOLING_BLOCKS:
            raise ValueError(f"지원하지 않는 특징 조합: {self.pooling}")
        feature_size = backbone.config.hidden_size * POOLING_BLOCKS[self.pooling]
        self.targets = [name for name, layer in backbone.named_modules()
                        if isinstance(layer, nn.Linear) and name.rsplit(".", 1)[-1] in ("q_proj", "v_proj", "query", "value")]
        if len(self.targets) != 24:
            raise ValueError(f"LoRA 적용 대상이 24개가 아님: {len(self.targets)}")
        self.backbone = get_peft_model(backbone, LoraConfig(
            r=config["lora_rank"], lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"],
            target_modules=self.targets, bias="none"))
        if checkpointing:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.head = nn.Linear(feature_size, 2)
        self.structure = {"model_type": backbone.config.model_type, "patch_size": self.patch_size,
                          "register_tokens": self.registers, "hidden_size": 768,
                          "patch_tokens": (self.input_size // self.patch_size)**2,
                          "lora_targets": self.targets, "pooling": self.pooling,
                          "head": [feature_size, 2],
                          "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
                          "total_parameters": sum(p.numel() for p in self.parameters())}

    def forward(self, pixels):
        if pixels.ndim != 4 or tuple(pixels.shape[1:]) != (3, self.input_size, self.input_size):
            raise ValueError(f"입력 텐서는 [batch, 3, {self.input_size}, {self.input_size}]여야 함")
        hidden = self.backbone(pixel_values=pixels).last_hidden_state
        if hidden.shape[1] != 1 + self.registers + (self.input_size // self.patch_size)**2:
            raise RuntimeError("예상한 CLS·register·패치 토큰 수와 다름")
        return self.head(pooled_features(hidden, self.registers, self.pooling))

    def trainable_state(self):
        return {"adapter": {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(self.backbone).items()},
                "head": {k: v.detach().cpu().clone() for k, v in self.head.state_dict().items()}}

    def load_trainable(self, state):
        if set(state["adapter"]) != set(get_peft_model_state_dict(self.backbone)):
            raise ValueError("LoRA 어댑터 키 불일치")
        result = set_peft_model_state_dict(self.backbone, state["adapter"])
        if result.unexpected_keys:
            raise ValueError(f"어댑터 키 불일치: {result.unexpected_keys}")
        self.head.load_state_dict(state["head"], strict=True)


def optimizer_for(model, config):
    return torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": config["lora_lr"]},
        {"params": model.head.parameters(), "lr": config["head_lr"]},
    ], weight_decay=config["weight_decay"])

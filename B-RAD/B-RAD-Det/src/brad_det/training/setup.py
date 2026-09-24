"""Model, anchor, and dataloader construction for training entry points."""

from __future__ import annotations

from torch.utils.data import DataLoader

from brad_det.datasets import BaseDetectionDataset, FewShotDetectionDataset
from brad_det.models import OwlVitDetectionHead, OwlVitModel, ViTBackbone
from brad_det.ops.anchors import generate_anchors


def build_model_and_anchors(config, device):
    model_config = config["model"]
    backbone_config = model_config["backbone"]
    detector_config = model_config["detector"]
    anchor_config = model_config["anchors"]
    input_size = backbone_config["input_size"]
    patch_size = backbone_config["patch_size"]
    num_classes = detector_config["base_num_classes"]
    num_classes += int(detector_config["predict_shadow_box"])
    num_classes += int(detector_config["predict_margin_box"])

    anchors, num_anchors = generate_anchors(
        feature_map_size=(input_size // patch_size, input_size // patch_size),
        stride=anchor_config["stride"],
        scales=anchor_config["scales"],
        aspect_ratios=anchor_config["aspect_ratios"],
        device=device,
    )
    anchors = anchors.unsqueeze(0).repeat(
        config["training"]["batch_size"], 1, 1, 1
    )
    backbone = ViTBackbone(
        model_name=backbone_config["name"],
        pretrained=backbone_config["pretrained"],
    )
    head = OwlVitDetectionHead(
        num_classes=num_classes,
        num_anchors=num_anchors,
        dropout=detector_config["dropout"],
    )
    model = OwlVitModel(
        backbone,
        head,
        num_classes=num_classes,
        input_size=input_size,
        patch_size=patch_size,
    ).to(device)
    return model, anchors, num_classes


def build_training_loaders(config, stage):
    training = config["training"]
    data = config["data"]
    detector = config["model"]["detector"]
    common = {
        "image_root": data["root"],
        "input_size": config["model"]["backbone"]["input_size"],
        "source_list": data.get("train_sources"),
        "long_bbox": detector["predict_shadow_box"],
        "margin_bbox_ratio": (
            detector["margin_box_ratio"]
            if detector["predict_margin_box"]
            else 0
        ),
    }
    if stage == "base":
        train_dataset = BaseDetectionDataset(
            data["train_metadata"],
            apply_aug=config["augmentation"]["enabled"],
            **common,
        )
        valid_dataset = BaseDetectionDataset(
            data["valid_metadata"],
            apply_aug=False,
            **common,
        )
    elif stage == "fewshot":
        reference_metadata = [
            data["valid_metadata"],
            data["ood_metadata"],
            data["train_metadata"],
            data["paper_crawl_metadata"],
        ]
        fewshot_common = {
            **common,
            "num_ref_shot": training["reference_shots"],
            "reference_metadata": reference_metadata,
        }
        train_dataset = FewShotDetectionDataset(
            data["train_metadata"],
            apply_aug=config["augmentation"]["enabled"],
            **fewshot_common,
        )
        valid_dataset = FewShotDetectionDataset(
            data["valid_metadata"],
            apply_aug=False,
            **fewshot_common,
        )
    else:
        raise ValueError(f"Unknown training stage: {stage}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=training["batch_size"],
        shuffle=True,
        num_workers=training["num_workers"],
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=training["num_workers"],
        pin_memory=True,
    )
    return train_loader, valid_loader

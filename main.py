import argparse
import csv
import logging
import math
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, random_split

from src.data_utils import get_data_real, get_hla_name_seq
from src.datasets import HLAIIDataset
from src.early_stopping import EarlyStopping
from src.networks import DSCA_HLAII


def parse_args():
    parser = argparse.ArgumentParser(description="Train DSCA-HLAII")
    parser.add_argument("--config", default="config/train.yaml", help="Path to YAML config file")
    parser.add_argument("--mode", choices=("train", "predict_core"), default="train")
    parser.add_argument("--checkpoint", help="Checkpoint path used by predict_core mode")
    parser.add_argument("--output", help="Output CSV path used by predict_core mode")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path) as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"配置文件 {config_path} 为空或格式非法")
    return config


def get_required(config, path):
    value = config
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"缺少配置项: {path}")
        value = value[key]
    return value


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_run_dir(config):
    output_root = get_required(config, "run.output_root")
    resume = bool(get_required(config, "run.resume"))
    run_name = config["run"].get("name") or datetime.now().strftime("%Y%m%d_%H%M%S")
    config["run"]["name"] = run_name

    run_dir = os.path.join(output_root, run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    last_ckpt_path = os.path.join(ckpt_dir, "last.pt")
    best_ckpt_path = os.path.join(ckpt_dir, "best.pt")

    os.makedirs(output_root, exist_ok=True)

    if resume:
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"resume=True，但运行目录不存在: {run_dir}")
        if not os.path.isfile(last_ckpt_path):
            raise FileNotFoundError(f"resume=True，但 last checkpoint 不存在: {last_ckpt_path}")
    else:
        if os.path.exists(run_dir) and os.listdir(run_dir):
            raise FileExistsError(f"运行目录已存在且非空，请更换 run.name 或启用 resume: {run_dir}")
        os.makedirs(ckpt_dir, exist_ok=True)

    os.makedirs(ckpt_dir, exist_ok=True)
    return {
        "run_dir": run_dir,
        "ckpt_dir": ckpt_dir,
        "last_ckpt_path": last_ckpt_path,
        "best_ckpt_path": best_ckpt_path,
        "metrics_path": os.path.join(run_dir, "metrics.csv"),
        "log_path": os.path.join(run_dir, "train.log"),
        "config_snapshot_path": os.path.join(run_dir, "config.snapshot.yaml"),
    }


def setup_logger(log_path):
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_path is not None:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def save_config_snapshot(config, snapshot_path):
    if os.path.exists(snapshot_path):
        return
    with open(snapshot_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


def build_optimizer(config, model):
    optimizer_name = get_required(config, "optimizer.name")
    lr = float(get_required(config, "optimizer.lr"))
    weight_decay = float(get_required(config, "optimizer.weight_decay"))

    if optimizer_name != "Adadelta":
        raise ValueError(f"当前仅支持 Adadelta，收到: {optimizer_name}")

    return torch.optim.Adadelta(model.parameters(), lr=lr, weight_decay=weight_decay)


def append_metrics(metrics_path, row):
    file_exists = os.path.exists(metrics_path)
    fieldnames = ["epoch", "train_loss", "valid_loss", "auroc"]

    with open(metrics_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def make_checkpoint_state(epoch, model, optimizer, early_stopping, metrics, run_name):
    return {
        "epoch": epoch,
        "run_name": run_name,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "early_stopping_state": early_stopping.state_dict(),
        "metrics": metrics,
    }


def save_checkpoint(path, state):
    torch.save(state, path)


def load_data_list(config):
    data_cfg = get_required(config, "data")
    hla_name_seq = get_hla_name_seq(data_cfg["hla_seq_file"])
    return get_data_real(
        hla_name_seq,
        data_cfg["train_file"],
        data_cfg["pep_esm_file"],
        data_cfg["hla_esm_file"],
        data_cfg["hla_esm_names_file"],
    )


def build_dataloaders(config):
    data_cfg = get_required(config, "data")
    batch_size = int(get_required(config, "train.batch_size"))
    valid_ratio = float(get_required(config, "train.valid_ratio"))
    seed = int(get_required(config, "run.seed"))

    data_list = load_data_list(config)
    dataset = HLAIIDataset(data_list)
    n_valid = int(len(dataset) * valid_ratio)
    n_train = len(dataset) - n_valid

    train_dataset, valid_dataset = random_split(
        dataset,
        [n_train, n_valid],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, n_train, n_valid


def train_main(config, args):
    paths = prepare_run_dir(config)
    logger = setup_logger(paths["log_path"])
    save_config_snapshot(config, paths["config_snapshot_path"])
    seed = int(get_required(config, "run.seed"))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用配置文件: {args.config}")
    logger.info(f"运行目录: {paths['run_dir']}")
    logger.info(f"使用设备: {device}")
    logger.info(f"随机种子: {seed}")

    train_loader, valid_loader, n_train, n_valid = build_dataloaders(config)
    logger.info(f"训练样本数: {n_train}")
    logger.info(f"验证样本数: {n_valid}")

    model = DSCA_HLAII().to(device)
    optimizer = build_optimizer(config, model)
    criterion = nn.BCELoss()
    early_stopping = EarlyStopping(
        patience=int(get_required(config, "train.patience")),
        verbose=True,
    )

    start_epoch = 0
    resume = bool(get_required(config, "run.resume"))
    if resume:
        logger.info(f"发现 checkpoint，从 {paths['last_ckpt_path']} 恢复...")
        ckpt = torch.load(paths["last_ckpt_path"], map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        early_stopping.load_state_dict(ckpt.get("early_stopping_state", {}))
        start_epoch = ckpt["epoch"] + 1
        metrics = ckpt.get("metrics", {})
        logger.info(
            "从 epoch %s 继续训练，上次指标: train_loss=%.4f valid_loss=%.4f auroc=%.4f",
            start_epoch,
            metrics.get("train_loss", float("nan")),
            metrics.get("valid_loss", float("nan")),
            metrics.get("auroc", float("nan")),
        )
    else:
        logger.info("未启用 resume，从头开始训练")

    num_epochs = int(get_required(config, "train.num_epochs"))
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        for (pep_oh, pep_esm, hla_oh, hla_esm), labels in train_loader:
            pep_oh = pep_oh.to(device)
            pep_esm = pep_esm.to(device)
            hla_oh = hla_oh.to(device)
            hla_esm = hla_esm.to(device)
            labels = labels.unsqueeze(1).to(device)

            optimizer.zero_grad()
            preds = model(pep_oh, pep_esm, hla_oh, hla_esm)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)

        train_loss /= n_train

        model.eval()
        valid_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for (pep_oh, pep_esm, hla_oh, hla_esm), labels in valid_loader:
                pep_oh = pep_oh.to(device)
                pep_esm = pep_esm.to(device)
                hla_oh = hla_oh.to(device)
                hla_esm = hla_esm.to(device)
                labels = labels.unsqueeze(1).to(device)

                preds = model(pep_oh, pep_esm, hla_oh, hla_esm)
                loss = criterion(preds, labels)
                valid_loss += loss.item() * len(labels)
                all_preds.extend(preds.squeeze(1).cpu().numpy())
                all_labels.extend(labels.squeeze(1).cpu().numpy())

        valid_loss /= n_valid
        auroc = roc_auc_score(all_labels, all_preds)
        logger.info(
            "Epoch %3d/%d | train_loss: %.4f | valid_loss: %.4f | AUROC: %.4f",
            epoch + 1,
            num_epochs,
            train_loss,
            valid_loss,
            auroc,
        )

        improved, early_stopping_message = early_stopping.step(valid_loss)
        if early_stopping_message:
            logger.info(early_stopping_message)

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": f"{train_loss:.6f}",
            "valid_loss": f"{valid_loss:.6f}",
            "auroc": f"{auroc:.6f}",
        }
        append_metrics(paths["metrics_path"], epoch_metrics)

        checkpoint_state = make_checkpoint_state(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            early_stopping=early_stopping,
            metrics={
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "auroc": auroc,
            },
            run_name=config["run"]["name"],
        )
        save_checkpoint(paths["last_ckpt_path"], checkpoint_state)
        if improved:
            save_checkpoint(paths["best_ckpt_path"], checkpoint_state)

        if early_stopping.early_stop:
            logger.info(f"Early stopping 触发，共训练 {epoch + 1} 个 epoch")
            break

    logger.info("训练完成")


def resolve_predict_paths(config, checkpoint_path, output_path):
    run_cfg = get_required(config, "run")
    run_name = run_cfg.get("name")
    output_root = get_required(config, "run.output_root")

    if checkpoint_path is None:
        if not run_name:
            raise ValueError("predict_core 模式下，如果未显式提供 --checkpoint，则 config.run.name 必须存在")
        checkpoint_path = os.path.join(output_root, run_name, "checkpoints", "best.pt")

    if output_path is None:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        run_dir = os.path.dirname(checkpoint_dir)
        output_path = os.path.join(run_dir, "core_predictions.csv")

    return checkpoint_path, output_path


def write_core_predictions(output_path, data_list, scores, core_positions):
    cutoff = 1.0 - math.log(500, 50000)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "PeptideSeq",
            "HLA_Alpha",
            "HLA_Beta",
            "Score",
            "Presentation",
            "CoreStart",
            "BindingCore",
        ])

        for sample, score, core_pos in zip(data_list, scores, core_positions):
            hla_name, peptide, _, _, _, _ = sample
            hla_alpha, hla_beta = hla_name.split("-", 1)
            score = float(score)
            core_pos = int(core_pos)
            presentation = "true" if score >= cutoff else "false"

            if presentation == "true" and peptide:
                core_pos = min(core_pos, len(peptide) - 1)
                binding_core = peptide[core_pos:core_pos + 9]
                core_start = core_pos
            else:
                binding_core = "NaN!"
                core_start = -1

            writer.writerow([
                peptide,
                hla_alpha,
                hla_beta,
                f"{score:.6f}",
                presentation,
                core_start,
                binding_core,
            ])


def predict_core_main(config, args):
    logger = setup_logger(None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path, output_path = resolve_predict_paths(config, args.checkpoint, args.output)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint_path}")

    seed = int(get_required(config, "run.seed"))
    set_seed(seed)

    logger.info(f"使用配置文件: {args.config}")
    logger.info(f"使用设备: {device}")
    logger.info(f"加载 checkpoint: {checkpoint_path}")

    data_list = load_data_list(config)
    batch_size = int(get_required(config, "train.batch_size"))
    data_loader = DataLoader(HLAIIDataset(data_list), batch_size=batch_size, shuffle=False)
    logger.info(f"预测样本数: {len(data_list)}")

    model = DSCA_HLAII().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_scores = []
    all_core_positions = []
    with torch.no_grad():
        for (pep_oh, pep_esm, hla_oh, hla_esm), _ in data_loader:
            pep_oh = pep_oh.to(device)
            pep_esm = pep_esm.to(device)
            hla_oh = hla_oh.to(device)
            hla_esm = hla_esm.to(device)

            scores = model(pep_oh, pep_esm, hla_oh, hla_esm, core=False).squeeze(1)
            core_scores = model(pep_oh, pep_esm, hla_oh, hla_esm, core=True)
            core_positions = core_scores.argmax(dim=-1)

            all_scores.extend(scores.cpu().numpy())
            all_core_positions.extend(core_positions.cpu().numpy())

    write_core_predictions(output_path, data_list, all_scores, all_core_positions)
    logger.info(f"预测结果已保存到: {output_path}")


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.mode == "train":
        train_main(config, args)
    else:
        predict_core_main(config, args)


if __name__ == "__main__":
    main()

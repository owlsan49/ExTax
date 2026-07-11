"""
训练脚本 - PersuRoleEmoNet
第一阶段: 多维特征对齐 (persu + role + emo)
第二阶段: 分类 (fake/real)
"""

import os
import sys
import argparse
import yaml
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import f1_score
from models.models import PersuRoleEmoNet

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import load_multi_feat_data, load_cls_data


def load_config(config_path):
    """加载YAML配置"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_epoch_align(model: nn.Module, dataloader: DataLoader,
                         criterion: nn.Module, optimizer: optim.Optimizer, device: str):
    """第一阶段训练: 多维特征对齐"""
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc="Training Align")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        persu_feat = batch['persu_feat'].to(device)
        role_feat = batch['role_feat'].to(device)
        emo_feat = batch['emo_feat'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, stage='align')
        multi_feat = outputs['multi_feat']  # [batch, 17]

        # 分割预测: persu(6) + role(6) + emo(5)
        persu_pred = multi_feat[:, 0:6]
        role_pred = multi_feat[:, 6:12]
        emo_pred = multi_feat[:, 12:]

        # 分别计算损失
        loss_persu = criterion(persu_pred, persu_feat)
        loss_role = criterion(role_pred, role_feat)
        loss_emo = criterion(emo_pred, emo_feat)

        loss = loss_persu + loss_role + loss_emo
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader)


def evaluate_align(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: str):
    """评估第一阶段模型"""
    model.eval()
    total_loss = 0
    all_persu_preds, all_persu_labels = [], []
    all_role_preds, all_role_labels = [], []
    all_emo_preds, all_emo_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Align"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            persu_feat = batch['persu_feat'].to(device)
            role_feat = batch['role_feat'].to(device)
            emo_feat = batch['emo_feat'].to(device)

            outputs = model(input_ids, attention_mask, stage='align')
            multi_feat = outputs['multi_feat']

            persu_pred = multi_feat[:, 0:6]
            role_pred = multi_feat[:, 6:12]
            emo_pred = multi_feat[:, 12:]

            loss_persu = criterion(persu_pred, persu_feat)
            loss_role = criterion(role_pred, role_feat)
            loss_emo = criterion(emo_pred, emo_feat)
            total_loss += (loss_persu + loss_role + loss_emo).item()
            
            persu_prob = torch.sigmoid(persu_pred)
            role_prob = torch.sigmoid(role_pred)
            emo_prob = torch.sigmoid(emo_pred)
            
            # 二值化用于计算F1
            all_persu_preds.append((persu_prob > 0.5).float().cpu().numpy())
            all_persu_labels.append(persu_feat.cpu().numpy())
            all_role_preds.append((role_prob > 0.5).float().cpu().numpy())
            all_role_labels.append(role_feat.cpu().numpy())
            all_emo_preds.append((emo_prob > 0.5).float().cpu().numpy())
            all_emo_labels.append(emo_feat.cpu().numpy())

    # 计算各任务F1 (对平滑后的连续label做0.5二值化后再计算)
    def calc_f1(preds: list, labels: list):
        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)
        labels = (labels > 0.5).astype(np.int64)
        return f1_score(labels, preds, average='macro', zero_division=0)

    return {
        'loss': total_loss / len(dataloader),
        'persu_f1': calc_f1(all_persu_preds, all_persu_labels) * 100,
        'role_f1': calc_f1(all_role_preds, all_role_labels) * 100,
        'emo_f1': calc_f1(all_emo_preds, all_emo_labels) * 100
    }


def train_epoch_cls(model: nn.Module, dataloader: DataLoader,
                    criterion: nn.Module, optimizer: optim.Optimizer, device: str):
    """第二阶段训练: 分类"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc="Training CLS")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, stage='cls')
        logits = outputs['cls_logits']

        loss = criterion(logits, labels)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average='macro') * 100

    return total_loss / len(dataloader), macro_f1


def evaluate_cls(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: str):
    """评估分类模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating CLS"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, stage='cls')
            logits = outputs['cls_logits']

            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    return {
        'loss': total_loss / len(dataloader),
        'macro_f1': macro_f1 * 100,
        'predictions': all_preds,
        'labels': all_labels
    }


def train(args):
    """训练主函数"""
    config = load_config(args.config)

    # 随机种子
    seed = config['train'].get('seed', 42)
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = config['train'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    stage = config['train']['stage']

    # 根据stage加载数据
    print("\n=== Loading Data ===")
    if stage == 'align':
        data_cfg = config['data']
        if 'models' in data_cfg and data_cfg['models']:
            train_loader, val_loader, test_loader = load_multi_feat_data(
                models=data_cfg['models'],
                train_datasets=data_cfg.get('train_datasets', []),
                val_datasets=data_cfg.get('val_datasets', []),
                test_datasets=data_cfg.get('test_datasets', []),
                data_dirs=data_cfg['data_dirs'],
                tokenizer_name=config['model']['encoder_name'],
                max_length=data_cfg['max_length'],
                batch_size=data_cfg['batch_size'],
                alpha=data_cfg.get('alpha', 0.5)
            )
        else:
            train_loader, val_loader, test_loader = load_multi_feat_data(
                train_files=data_cfg.get('train_files'),
                val_files=data_cfg.get('val_files'),
                test_files=data_cfg.get('test_files'),
                data_dirs=data_cfg['data_dirs'],
                tokenizer_name=config['model']['encoder_name'],
                max_length=data_cfg['max_length'],
                batch_size=data_cfg['batch_size']
            )
    else:  # cls
        data_cfg = config['data']
        if 'models' in data_cfg and data_cfg['models']:
            model = data_cfg['models'][0]
            split_map = {'train': 'train', 'val': 'validation', 'test': 'test'}
            train_files = [f"{model}_{ds}_{split_map['train']}.csv" for ds in data_cfg.get('train_datasets', [])]
            val_files = [f"{model}_{ds}_{split_map['val']}.csv" for ds in data_cfg.get('val_datasets', [])]
            test_files = [f"{model}_{ds}_{split_map['test']}.csv" for ds in data_cfg.get('test_datasets', [])]
        else:
            train_files = data_cfg.get('train_files')
            val_files = data_cfg.get('val_files')
            test_files = data_cfg.get('test_files')

        train_loader, val_loader, test_loader = load_cls_data(
            train_files=train_files,
            val_files=val_files,
            test_files=test_files,
            data_dirs=data_cfg['data_dirs'],
            tokenizer_name=config['model']['encoder_name'],
            max_length=data_cfg['max_length'],
            batch_size=data_cfg['batch_size']
        )

    # 构建模型
    print("\n=== Building Model ===")
    model = PersuRoleEmoNet(
        encoder_name=config['model']['encoder_name'],
        pooling=config['model']['pooling'],
        persu_hidden=config['model']['persu_hidden'],
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout'],
        max_length=config['data']['max_length'],
        num_learnable_queries=config['model']['num_learnable_queries'],
        num_hetero_attention=config['model'].get('num_hetero_attention', 3)
    ).to(device)

    # 加载预训练模型
    pretrained_path = config['train'].get('pretrained_path')
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained model from {pretrained_path}...")
        state_dict = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print("Pretrained model loaded!")
    else:
        if pretrained_path:
            print(f"Pretrained path specified but file not found: {pretrained_path}")
        print("Training from scratch")

    # 设置可训练参数
    model.set_trainable(stage)
    print(f"Trainable params: {model.get_trainable_params()} / {model.get_total_params()}")

    # 优化器和损失函数
    criterion = nn.BCEWithLogitsLoss() if stage == 'align' else nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['train']['learning_rate'],
        weight_decay=config['train']['weight_decay']
    )

    # 训练
    best_val_f1 = 0
    patience_counter = 0
    best_model_state = None

    if stage == 'align':
        print("\n=== Training Stage 1: Multi-Feature Alignment ===")
        for epoch in range(config['train']['epochs']):
            train_loss = train_epoch_align(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate_align(model, val_loader, criterion, device)

            print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | "
                  f"Persu F1: {val_metrics['persu_f1']:.2f}% | "
                  f"Role F1: {val_metrics['role_f1']:.2f}% | "
                  f"Emo F1: {val_metrics['emo_f1']:.2f}%")

            avg_f1 = (val_metrics['persu_f1'] + val_metrics['role_f1'] + val_metrics['emo_f1']) / 3
            if avg_f1 > best_val_f1:
                best_val_f1 = avg_f1
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config['train']['patience']:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

        if best_model_state:
            model.load_state_dict(best_model_state)
            model = model.to(device)

        print("\n=== Test Results ===")
        test_metrics = evaluate_align(model, test_loader, criterion, device)
        print(f"Test Loss: {test_metrics['loss']:.4f}")
        print(f"Test Persu F1: {test_metrics['persu_f1']:.2f}%")
        print(f"Test Role F1: {test_metrics['role_f1']:.2f}%")
        print(f"Test Emo F1: {test_metrics['emo_f1']:.2f}%")

    else:  # cls
        print("\n=== Training Stage 2: Classification ===")
        for epoch in range(config['train']['epochs']):
            train_loss, train_f1 = train_epoch_cls(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate_cls(model, val_loader, criterion, device)

            print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | "
                  f"Train Macro F1: {train_f1:.2f}% | "
                  f"Val Macro F1: {val_metrics['macro_f1']:.2f}%")

            if val_metrics['macro_f1'] > best_val_f1:
                best_val_f1 = val_metrics['macro_f1']
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config['train']['patience']:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

        if best_model_state:
            model.load_state_dict(best_model_state)
            model = model.to(device)

        print("\n=== Test Results ===")
        test_metrics = evaluate_cls(model, test_loader, criterion, device)
        print(f"Test Loss: {test_metrics['loss']:.4f}")
        print(f"Test Macro F1: {test_metrics['macro_f1']:.2f}%")

    # 保存模型
    if config['save'].get('save_path'):
        os.makedirs(os.path.dirname(config['save']['save_path']), exist_ok=True)
        torch.save(model.state_dict(), config['save']['save_path'])
        print(f"\nModel saved to {config['save']['save_path']}")

    return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PersuRoleEmoNet")
    parser.add_argument("--config", type=str, default="config/stage1.yaml", help="Config file path")
    args = parser.parse_args()

    train(args)
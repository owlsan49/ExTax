"""
数据加载模块 - PersuRoleEmoNet
两个数据集:
1. MultiFeatDataset: text + persu_feat + role_feat + emo_feat (第一阶段)
2. ClsDataset: text + label (第二阶段分类)
"""

import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List
from transformers import AutoTokenizer


PERSUASION_FEATURES = [
    'Attack_on_reputation',
    'Justification',
    'Simplification',
    'Distraction',
    'Call',
    'Manipulative_wording'
]

ROLE_FEATURES = [
    'Ethical_Stabilizers',
    'Altruistic_Catalysts',
    'Overt_Aggressors',
    'Deceptive_Subversives',
    'Institutional_Toxins',
    'Marginalized_Sufferers'
]

EMO_FEATURES = [
    'Fear',
    'Anger',
    'Hope',
    'Anxiety',
    'Sadness'
]


def parse_features(json_str, feature_names):
    """解析特征 - 将JSON转换为n维向量"""
    dim = len(feature_names)
    if pd.isna(json_str) or not isinstance(json_str, str):
        return np.zeros(dim, dtype=np.float32)

    json_str = json_str.strip()
    # 处理某些模型输出的 {{ }} 格式
    if json_str.startswith('{{'):
        json_str = json_str.replace('{{', '{').replace('}}', '}')

    try:
        data = json.loads(json_str)
    except:
        try:
            data = json.loads(json_str.replace("'", '"'))
        except:
            return np.zeros(dim, dtype=np.float32)

    if not isinstance(data, dict):
        return np.zeros(dim, dtype=np.float32)

    features = []
    for feature in feature_names:
        if feature in data:
            val = data[feature]
            if isinstance(val, dict):
                is_used = val.get('is_used', 'No')
            elif isinstance(val, str):
                is_used = val
            else:
                is_used = 'No'
            features.append(1.0 if str(is_used).strip().lower() == 'yes' else 0.0)
        else:
            features.append(0.0)
    return np.array(features, dtype=np.float32)


def compute_smoothed_labels(model_preds, alpha=0.5):
    """
    基于多模型预测计算平滑后的标签值。

    公式:
        p_{i,c} = mean_m(pred_{m,i,c})
        H_{i,c} = -[p_{i,c} log2(p_{i,c}) + (1-p_{i,c}) log2(1-p_{i,c})]
        y_{i,c} = p_{i,c} * (1 - alpha * H_{i,c}) + 0.5 * (alpha * H_{i,c})

    Args:
        model_preds: np.ndarray, shape [num_models, num_samples, num_features]
        alpha: 平滑超参数，默认 0.5

    Returns:
        np.ndarray, shape [num_samples, num_features]
    """
    p = model_preds.mean(axis=0)  # [num_samples, num_features]

    # 计算熵，处理边界情况
    H = np.zeros_like(p)
    mask = (p > 0) & (p < 1)
    p_masked = p[mask]
    H[mask] = -(p_masked * np.log2(p_masked) + (1 - p_masked) * np.log2(1 - p_masked))

    smoothed = p * (1 - alpha * H) + 0.5 * (alpha * H)
    return smoothed.astype(np.float32)


class MultiFeatDataset(Dataset):
    """第一阶段数据集: text + persu_feat + role_feat + emo_feat"""

    def __init__(
        self,
        texts: List[str],
        persu_feats: np.ndarray,
        role_feats: np.ndarray,
        emo_feats: np.ndarray,
        tokenizer,
        max_length: int = 512
    ):
        self.texts = texts
        self.persu_feats = torch.tensor(persu_feats, dtype=torch.float32)
        self.role_feats = torch.tensor(role_feats, dtype=torch.float32)
        self.emo_feats = torch.tensor(emo_feats, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'persu_feat': self.persu_feats[idx],
            'role_feat': self.role_feats[idx],
            'emo_feat': self.emo_feats[idx]
        }


class ClsDataset(Dataset):
    """第二阶段数据集: text + label (fake/real分类)"""

    def __init__(
        self,
        texts: List[str],
        labels: np.ndarray,
        tokenizer,
        max_length: int = 512
    ):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': self.labels[idx]
        }


def load_multi_feat_data(
    train_files: List[str] = None,
    val_files: List[str] = None,
    test_files: List[str] = None,
    models: List[str] = None,
    train_datasets: List[str] = None,
    val_datasets: List[str] = None,
    test_datasets: List[str] = None,
    data_dirs: List[str] = None,
    tokenizer_name: str = "roberta-base",
    max_length: int = 512,
    batch_size: int = 16,
    alpha: float = 0.5
):
    """
    加载多维特征数据 (第一阶段)

    支持两种模式:
    1. 旧模式: 直接指定 train_files/val_files/test_files 文件名列表
    2. 新模式: 指定 models 和 train_datasets/val_datasets/test_datasets，
       自动构造文件名 {model}_{dataset}_{split}.csv，并对多模型预测做标签平滑

    Args:
        train_files/val_files/test_files: 旧模式文件名列表
        models: 新模式 - 模型名称列表
        train_datasets/val_datasets/test_datasets: 新模式 - 数据集名称列表
        data_dirs: 数据目录列表，按顺序对应 persu/role/emo 数据
        alpha: 标签平滑超参数，默认 0.5

    Returns:
        train_loader, val_loader, test_loader
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # data_dirs 顺序: [persu_dir, role_dir, emo_dir]
    persu_dir = data_dirs[0]
    role_dir = data_dirs[1]
    emo_dir = data_dirs[2]

    use_new_mode = models is not None

    if not use_new_mode:
        # ===== 旧模式 =====
        all_data = []
        splits = {'train': train_files, 'val': val_files, 'test': test_files}

        for split, files in splits.items():
            if not files:
                continue
            for f in files:
                base_path = os.path.join(persu_dir, f)
                if os.path.exists(base_path):
                    df = pd.read_csv(base_path)
                    df['split'] = split
                    df['data_source'] = 'persu'

                    role_path = os.path.join(role_dir, f)
                    emo_path = os.path.join(emo_dir, f)

                    if os.path.exists(role_path):
                        df_role = pd.read_csv(role_path)
                        if 'generated_pred' in df_role.columns:
                            df['role_pred'] = df_role['generated_pred']

                    if os.path.exists(emo_path):
                        df_emo = pd.read_csv(emo_path)
                        if 'generated_pred' in df_emo.columns:
                            df['emo_pred'] = df_emo['generated_pred']

                    all_data.append(df)
                    print(f"Loaded {split}: {len(df)} rows from {f}")

        if not all_data:
            raise ValueError("No data loaded!")

        df_all = pd.concat(all_data, ignore_index=True)

        texts = df_all['content'].fillna('').tolist()
        persu_feats = np.array([parse_features(row.get('generated_pred', ''), PERSUASION_FEATURES)
                                for _, row in df_all.iterrows()], dtype=np.float32)
        role_feats = np.array([parse_features(row.get('role_pred', ''), ROLE_FEATURES)
                              for _, row in df_all.iterrows()], dtype=np.float32)
        emo_feats = np.array([parse_features(row.get('emo_pred', ''), EMO_FEATURES)
                             for _, row in df_all.iterrows()], dtype=np.float32)

        train_mask = df_all['split'] == 'train'
        val_mask = df_all['split'] == 'val'
        test_mask = df_all['split'] == 'test'

    else:
        # ===== 新模式: 多模型标签平滑 =====
        split_datasets = {
            'train': train_datasets or [],
            'val': val_datasets or [],
            'test': test_datasets or []
        }
        split_file_map = {'train': 'train', 'val': 'validation', 'test': 'test'}

        all_texts = []
        all_persu_feats = []
        all_role_feats = []
        all_emo_feats = []
        all_splits = []

        for split, datasets in split_datasets.items():
            split_file = split_file_map[split]
            for dataset in datasets:
                texts = None
                model_persu_preds = []
                model_role_preds = []
                model_emo_preds = []
                valid_models = []

                base_df = None
                align_col = None

                for model in models:
                    filename = f"{model}_{dataset}_{split_file}.csv"

                    persu_path = os.path.join(persu_dir, filename)
                    role_path = os.path.join(role_dir, filename)
                    emo_path = os.path.join(emo_dir, filename)

                    if not os.path.exists(persu_path):
                        print(f"Warning: {persu_path} not found, skipping model {model} for {dataset}/{split}")
                        continue

                    df_persu = pd.read_csv(persu_path)

                    if texts is None:
                        texts = df_persu['content'].fillna('').tolist()
                        base_df = df_persu
                        if 'uuid' in df_persu.columns:
                            align_col = 'uuid'
                        else:
                            align_col = None
                    else:
                        if align_col is not None and align_col in df_persu.columns:
                            df_persu = df_persu.set_index(align_col).loc[base_df[align_col]].reset_index()
                        elif len(df_persu) == len(base_df):
                            pass
                        else:
                            df_persu = df_persu.set_index('content').loc[base_df['content']].reset_index()

                    persu_preds = np.array([parse_features(row, PERSUASION_FEATURES)
                                           for row in df_persu['generated_pred']])
                    model_persu_preds.append(persu_preds)

                    if os.path.exists(role_path):
                        df_role = pd.read_csv(role_path)
                        if align_col is not None and align_col in df_role.columns:
                            df_role = df_role.set_index(align_col).loc[base_df[align_col]].reset_index()
                        elif len(df_role) == len(base_df):
                            pass
                        else:
                            df_role = df_role.set_index('content').loc[base_df['content']].reset_index()
                        role_preds = np.array([parse_features(row, ROLE_FEATURES)
                                              for row in df_role['generated_pred']])
                        model_role_preds.append(role_preds)
                    else:
                        model_role_preds.append(np.zeros((len(texts), len(ROLE_FEATURES))))

                    if os.path.exists(emo_path):
                        df_emo = pd.read_csv(emo_path)
                        if align_col is not None and align_col in df_emo.columns:
                            df_emo = df_emo.set_index(align_col).loc[base_df[align_col]].reset_index()
                        elif len(df_emo) == len(base_df):
                            pass
                        else:
                            df_emo = df_emo.set_index('content').loc[base_df['content']].reset_index()
                        emo_preds = np.array([parse_features(row, EMO_FEATURES)
                                             for row in df_emo['generated_pred']])
                        model_emo_preds.append(emo_preds)
                    else:
                        model_emo_preds.append(np.zeros((len(texts), len(EMO_FEATURES))))

                    valid_models.append(model)

                if not model_persu_preds:
                    print(f"Warning: No model data found for {dataset}/{split}")
                    continue

                # Stack and compute smoothed labels
                persu_stack = np.stack(model_persu_preds, axis=0)  # [M, N, C]
                role_stack = np.stack(model_role_preds, axis=0)
                emo_stack = np.stack(model_emo_preds, axis=0)

                persu_smooth = compute_smoothed_labels(persu_stack, alpha)
                role_smooth = compute_smoothed_labels(role_stack, alpha)
                emo_smooth = compute_smoothed_labels(emo_stack, alpha)

                all_texts.extend(texts)
                all_persu_feats.append(persu_smooth)
                all_role_feats.append(role_smooth)
                all_emo_feats.append(emo_smooth)
                all_splits.extend([split] * len(texts))
                print(f"Loaded {split}/{dataset}: {len(texts)} rows, {len(valid_models)} models, alpha={alpha}")

        if not all_texts:
            raise ValueError("No data loaded!")

        texts = all_texts
        persu_feats = np.concatenate(all_persu_feats, axis=0)
        role_feats = np.concatenate(all_role_feats, axis=0)
        emo_feats = np.concatenate(all_emo_feats, axis=0)

        splits_arr = np.array(all_splits)
        train_mask = splits_arr == 'train'
        val_mask = splits_arr == 'val'
        test_mask = splits_arr == 'test'

    # 创建数据集 (新旧模式通用)
    train_dataset = MultiFeatDataset(
        [t for m, t in zip(train_mask, texts) if m],
        persu_feats[train_mask],
        role_feats[train_mask],
        emo_feats[train_mask],
        tokenizer,
        max_length
    )
    val_dataset = MultiFeatDataset(
        [t for m, t in zip(val_mask, texts) if m],
        persu_feats[val_mask],
        role_feats[val_mask],
        emo_feats[val_mask],
        tokenizer,
        max_length
    )
    test_dataset = MultiFeatDataset(
        [t for m, t in zip(test_mask, texts) if m],
        persu_feats[test_mask],
        role_feats[test_mask],
        emo_feats[test_mask],
        tokenizer,
        max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Total: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    return train_loader, val_loader, test_loader


def load_cls_data(
    train_files: List[str] = None,
    val_files: List[str] = None,
    test_files: List[str] = None,
    data_dirs: List[str] = None,
    tokenizer_name: str = "roberta-base",
    max_length: int = 512,
    batch_size: int = 16
):
    """
    加载分类数据 (第二阶段)

    Args:
        train_files/val_files/test_files: 文件名列表
        data_dirs: 数据目录列表

    Returns:
        train_loader, val_loader, test_loader
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    all_data = []
    splits = {'train': train_files, 'val': val_files, 'test': test_files}

    for split, files in splits.items():
        if not files:
            continue
        for f in files:
            for data_dir in data_dirs:
                path = os.path.join(data_dir, f)
                print(path)
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    df['split'] = split
                    all_data.append(df)
                    print(f"Loaded {split}: {len(df)} from {f}")
                    break

    if not all_data:
        raise ValueError("No data loaded!")

    df_all = pd.concat(all_data, ignore_index=True)

    # 解析文本和标签 (fake=0, real=1)
    texts = df_all['content'].fillna('').tolist()
    labels = df_all['label'].map({'fake': 0, 'real': 1}).fillna(0).astype(np.int64).values

    # 按split划分
    train_mask = df_all['split'] == 'train'
    val_mask = df_all['split'] == 'val'
    test_mask = df_all['split'] == 'test'

    # 创建数据集
    train_dataset = ClsDataset(
        [t for m, t in zip(train_mask, texts) if m],
        labels[train_mask],
        tokenizer,
        max_length
    )
    val_dataset = ClsDataset(
        [t for m, t in zip(val_mask, texts) if m],
        labels[val_mask],
        tokenizer,
        max_length
    )
    test_dataset = ClsDataset(
        [t for m, t in zip(test_mask, texts) if m],
        labels[test_mask],
        tokenizer,
        max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Total: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    return train_loader, val_loader, test_loader
"""
Simple evaluation script without adaptation.
Just loads the source model and evaluates on corruption datasets.
Uses per-frame mIoU calculation (same as adapt_online.py).
"""
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import jaccard_score
import MinkowskiEngine as ME
from prettytable import PrettyTable
import csv

import models
from utils.config import get_config
from utils.collation import CollateFN
from utils.dataset_online import get_online_dataset, FrameOnlineDataset

np.random.seed(1234)
torch.manual_seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, required=True)
parser.add_argument("--split_size", default=4071, type=int)


def load_checkpoint(model, checkpoint_path, device):
    """Load model checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Clean state dict from PL names
    state_dict = {}
    for k, v in ckpt['state_dict'].items():
        if 'model.' in k:
            new_k = k.replace('model.', '')
            state_dict[new_k] = v
        else:
            state_dict[k] = v

    model.load_state_dict(state_dict)
    return model


def compute_frame_iou(preds, labels, num_classes, ignore_label=-1):
    """Compute IoU per class for a single frame."""
    valid_mask = labels != ignore_label
    preds = preds[valid_mask]
    labels = labels[valid_mask]

    if len(preds) == 0:
        return np.zeros(num_classes)

    iou_per_class = jaccard_score(labels, preds, average=None,
                                   labels=list(range(num_classes)),
                                   zero_division=0)
    # Set IoU=1 to 0 (same as adapt_online.py)
    iou_per_class[iou_per_class == 1] = 0
    return iou_per_class


def evaluate(config, args):
    device = torch.device(f'cuda:{config.pipeline.gpu}')

    # Create dataset
    mapping_path = config.dataset.mapping_path
    corrupt_list = config.dataset.corrupt_type
    level = config.dataset.level

    dataset = get_online_dataset(
        dataset_name=config.dataset.name,
        dataset_path=config.dataset.dataset_path,
        voxel_size=config.dataset.voxel_size,
        augment_data=False,
        max_time_wdw=config.dataset.max_time_window,
        version=config.dataset.version,
        sub_num=config.dataset.num_pts,
        ignore_label=config.dataset.ignore_label,
        split_size=args.split_size,
        mapping_path=mapping_path,
        num_classes=config.model.out_classes,
        corrupt_list=corrupt_list
    )

    # Create model
    Model = getattr(models, config.model.name)
    model = Model(config.model.in_feat_size, config.model.out_classes)
    model = load_checkpoint(model, config.pipeline.source_model, device)
    model = model.to(device)
    model.eval()

    num_classes = config.model.out_classes
    ignore_label = config.dataset.ignore_label

    # Collate function (use CPU, move to GPU later)
    collate_fn = CollateFN()
    collate_fn.device = torch.device('cpu')

    # Results storage
    all_results = {}

    # Class names for display
    if num_classes == 7:
        class_names = ['Vehicle', 'Pedestrian', 'Road', 'Sidewalk', 'Terrain', 'Manmade', 'Vegetation']
    elif num_classes == 3:
        class_names = ['Background', 'Vehicle', 'Pedestrian']
    else:
        class_names = ['Vehicle', 'Pedestrian']

    # Evaluate each corruption type
    num_sequences = dataset.num_sequences()

    print("\n" + "=" * 60)
    print(f"Evaluation (No Adaptation) - Level: {level}")
    print(f"Corruptions: {corrupt_list}")
    print("=" * 60 + "\n")

    for seq_idx in range(num_sequences):
        corrupt_name = corrupt_list[seq_idx]
        dataset.set_sequence(seq_idx)

        frame_dataset = FrameOnlineDataset(dataset)
        dataloader = DataLoader(
            frame_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config.pipeline.dataloader.num_workers,
            collate_fn=collate_fn
        )

        # Per-frame IoU storage
        frame_ious = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f'[{seq_idx+1}/{num_sequences}] {corrupt_name}'):
                coords = batch['coordinates'].int().to(device)
                feats = batch['features'].to(device)
                labels = batch['labels'].numpy()

                # Create sparse tensor
                stensor = ME.SparseTensor(
                    coordinates=coords,
                    features=feats,
                    quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE
                )

                # Forward pass
                output = model(stensor)
                if isinstance(output, tuple):
                    output = output[0]

                # Get predictions
                if hasattr(output, 'F'):
                    logits = output.F
                else:
                    logits = output

                preds = logits.argmax(dim=1).cpu().numpy()

                # Compute per-frame IoU (same as adapt_online.py)
                frame_iou = compute_frame_iou(preds, labels, num_classes, ignore_label)
                frame_ious.append(frame_iou)

        # Compute per-class mean IoU across frames
        frame_ious = np.array(frame_ious)  # [num_frames, num_classes]
        per_class_iou = np.nanmean(frame_ious, axis=0)
        miou = np.nanmean(per_class_iou)

        all_results[corrupt_name] = {
            'iou_per_class': per_class_iou,
            'miou': miou
        }

        print(f"\n>>> [{corrupt_name}] mIoU: {miou*100:.2f}%\n")

    # Create and print summary table
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS SUMMARY (No Adaptation)")
    print("=" * 80)

    table = PrettyTable()
    table.field_names = ['Corruption', 'mIoU'] + class_names

    all_miou = []
    all_per_class = []
    for corrupt_name, result in all_results.items():
        miou = result['miou'] * 100
        per_class = result['iou_per_class'] * 100
        all_miou.append(miou)
        all_per_class.append(per_class)

        row = [corrupt_name, f'{miou:.2f}']
        row.extend([f'{iou:.2f}' for iou in per_class])
        table.add_row(row)

    # Add average row
    avg_miou = np.mean(all_miou)
    avg_per_class = np.mean(all_per_class, axis=0)
    avg_row = ['Average', f'{avg_miou:.2f}']
    avg_row.extend([f'{iou:.2f}' for iou in avg_per_class])
    table.add_row(avg_row)

    table.align = 'r'
    table.align['Corruption'] = 'l'
    print(table)
    print(f"\nAverage mIoU: {avg_miou:.2f}%")

    # Save results
    save_dir = config.pipeline.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # Save as txt
    txt_path = os.path.join(save_dir, 'eval_summary_results.txt')
    with open(txt_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EVALUATION RESULTS SUMMARY (No Adaptation)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Level: {level}\n")
        f.write(f"Number of corruptions: {len(all_results)}\n\n")
        f.write(str(table))
        f.write("\n\n")
        f.write(f"Average mIoU: {avg_miou:.2f}%\n")
    print(f"\nSummary table saved to: {txt_path}")

    # Save as csv
    csv_path = os.path.join(save_dir, 'eval_summary_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Corruption', 'mIoU'] + class_names)
        for corrupt_name, result in all_results.items():
            miou = result['miou'] * 100
            per_class = result['iou_per_class'] * 100
            row = [corrupt_name, f'{miou:.2f}']
            row.extend([f'{iou:.2f}' for iou in per_class])
            writer.writerow(row)
        avg_row = ['Average', f'{avg_miou:.2f}']
        avg_row.extend([f'{iou:.2f}' for iou in avg_per_class])
        writer.writerow(avg_row)
    print(f"Summary CSV saved to: {csv_path}")

    # Save raw results
    npy_path = os.path.join(save_dir, 'eval_results.npy')
    np.save(npy_path, all_results)
    print(f"Raw results saved to: {npy_path}")

    return all_results


if __name__ == '__main__':
    args = parser.parse_args()
    config = get_config(args.config_file)
    evaluate(config, args)

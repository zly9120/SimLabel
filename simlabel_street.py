import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import os
from PIL import Image
import numpy as np
from sklearn.metrics import cohen_kappa_score
from torch.optim import AdamW
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
import random
from timechat.models.blip2 import Blip2Base, disabled_train
from timechat.models.Qformer import BertConfig, BertLMHeadModel
import einops
import torch.nn.init as init

MAIN_CATEGORIES = ['Happiness', 'Healthy', 'Safe', 'Lively', 'Orderly']
NUM_LEVELS = 6

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class ImageDataset(Dataset):
    def __init__(self, img_dir, csv_file, category, transform=None, missing_rate=0.0):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.category = category
        self.missing_rate = missing_rate
        self.num_annotators = 10

        if self.missing_rate > 0:
            self.simulate_missing_labels()

    def simulate_missing_labels(self):
        np.random.seed(42)
        n_samples = len(self.annotations)

        for i in range(1, self.num_annotators + 1):
            col_name = f"{self.category}_{i}"
            mask = np.random.random(n_samples) < self.missing_rate
            self.annotations.loc[mask, col_name] = -1

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        labels = []
        valid_mask = []
        for i in range(1, self.num_annotators + 1):
            label = self.annotations[f"{self.category}_{i}"].iloc[idx]
            if label == -1:
                labels.append(0)
                valid_mask.append(False)
            else:
                labels.append(label)
                valid_mask.append(True)

        labels = torch.tensor(labels, dtype=torch.long)
        valid_mask = torch.tensor(valid_mask, dtype=torch.bool)
        return image, labels, valid_mask


class SimilarityManager:
    def __init__(self, num_annotators, num_levels, device):
        self.num_annotators = num_annotators
        self.num_levels = num_levels
        self.device = device
        self.similarity_matrix = None
        self.annotation_counts = torch.zeros((num_annotators, num_annotators), device=device)
        self.confusion_matrices = torch.zeros((num_annotators, num_annotators, num_levels, num_levels), device=device)

    def update_from_batch(self, labels, valid_masks):
        batch_size = labels.size(0)

        for i in range(self.num_annotators):
            for j in range(i + 1, self.num_annotators):
                valid_both = valid_masks[:, i] & valid_masks[:, j]

                if valid_both.any():
                    labels_i = labels[valid_both, i]
                    labels_j = labels[valid_both, j]

                    for k in range(valid_both.sum()):
                        self.confusion_matrices[i, j, labels_i[k], labels_j[k]] += 1
                        self.confusion_matrices[j, i, labels_j[k], labels_i[k]] += 1

                    self.annotation_counts[i, j] += valid_both.sum()
                    self.annotation_counts[j, i] += valid_both.sum()

    def calculate_similarity_matrix(self):
        self.similarity_matrix = torch.eye(self.num_annotators, device=self.device)

        for i in range(self.num_annotators):
            for j in range(i + 1, self.num_annotators):
                if self.annotation_counts[i, j] > 0:
                    n_samples = self.annotation_counts[i, j]
                    confusion = self.confusion_matrices[i, j]

                    observed_agreement = torch.diag(confusion).sum() / n_samples

                    row_sum = confusion.sum(dim=1)
                    col_sum = confusion.sum(dim=0)
                    expected_agreement = (row_sum * col_sum).sum() / (n_samples * n_samples)

                    if expected_agreement < 1.0:
                        kappa = (observed_agreement - expected_agreement) / (1.0 - expected_agreement)
                        kappa = max(0, kappa.item())
                    else:
                        kappa = 1.0

                    self.similarity_matrix[i, j] = kappa
                    self.similarity_matrix[j, i] = kappa

        return self.similarity_matrix

    def get_similarity_weights(self, annotator_idx):
        if self.similarity_matrix is None:
            self.calculate_similarity_matrix()

        weights = self.similarity_matrix[annotator_idx].clone()
        weights[annotator_idx] = 0

        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = torch.ones(self.num_annotators, device=self.device) / (self.num_annotators - 1)
            weights[annotator_idx] = 0

        return weights


class Backbone(Blip2Base):
    def __init__(
            self,
            vit_model="eva_vit_g.pth",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            multi_annotation=True,
            q_former_model="instruct_blip_vicuna7b_trimmed.pth",
            freeze_qformer=False,
            num_query_token=10,
            attention_dim=256,
            num_levels=6,
            num_annotators=10
    ):
        super().__init__()

        self.num_annotators = num_annotators
        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )

        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.Qformer.cls = None

        self.load_from_pretrained(multi_annotation=multi_annotation, url_or_filename=q_former_model)

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, attention_dim)
        self.dropout = nn.Dropout(0.5)
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(attention_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_levels)
            ) for _ in range(num_annotators)
        ])

    def forward(self, images):
        device = images.device
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(images)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                output_attentions=True,
                return_dict=True,
            )
            frame_cross_attentions = query_output.cross_attentions[-2]
            frame_cross_attentions = frame_cross_attentions[:, :, :, 1:]
            B, H, Q, L = frame_cross_attentions.shape
            frame_cross_attentions = frame_cross_attentions.reshape(B, H, Q, int(np.sqrt(L)), int(np.sqrt(L)))

            image_hidden = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_hidden), dim=-1)

        features = self.dropout(image_features)

        outputs = []
        for annotator_idx in range(self.num_annotators):
            annotator_feature = features[:, annotator_idx, :]
            classifier = self.output_heads[annotator_idx]
            outputs.append(classifier(annotator_feature))

        outputs = torch.stack(outputs, dim=1)
        return outputs, frame_cross_attentions


class SimLabelTrainer:
    def __init__(self, num_annotators, num_levels, confidence_threshold=0.6, device='cuda'):
        self.num_annotators = num_annotators
        self.num_levels = num_levels
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.similarity_manager = SimilarityManager(num_annotators, num_levels, device)
        self.imputed_labels = {}

    def generate_soft_labels(self, outputs, valid_masks):
        batch_size = outputs.size(0)
        soft_labels = []
        confidence_scores = []

        for sample_idx in range(batch_size):
            sample_soft_labels = []
            sample_confidences = []

            for annotator_idx in range(self.num_annotators):
                if valid_masks[sample_idx, annotator_idx]:
                    sample_soft_labels.append(None)
                    sample_confidences.append(None)
                else:
                    weights = self.similarity_manager.get_similarity_weights(annotator_idx)
                    weighted_preds = torch.zeros(self.num_levels, device=self.device)

                    for k in range(self.num_annotators):
                        if k != annotator_idx and valid_masks[sample_idx, k]:
                            pred_probs = F.softmax(outputs[sample_idx, k], dim=0)
                            weighted_preds += weights[k] * pred_probs

                    if weighted_preds.sum() > 0:
                        weighted_preds = weighted_preds / weighted_preds.sum()

                        max_prob = weighted_preds.max()
                        entropy = -(weighted_preds * torch.log(weighted_preds + 1e-8)).sum()
                        max_entropy = math.log(self.num_levels)
                        confidence = max_prob * (1 - entropy / max_entropy)

                        sample_soft_labels.append(weighted_preds)
                        sample_confidences.append(confidence)
                    else:
                        sample_soft_labels.append(None)
                        sample_confidences.append(None)

            soft_labels.append(sample_soft_labels)
            confidence_scores.append(sample_confidences)

        return soft_labels, confidence_scores

    def update_imputed_labels(self, batch_idx, soft_labels, confidence_scores):
        for sample_idx, (sample_soft_labels, sample_confidences) in enumerate(zip(soft_labels, confidence_scores)):
            for annotator_idx, (soft_label, confidence) in enumerate(zip(sample_soft_labels, sample_confidences)):
                if soft_label is not None and confidence is not None and confidence > self.confidence_threshold:
                    pred_label = torch.argmax(soft_label).item()
                    key = (batch_idx, sample_idx, annotator_idx)
                    self.imputed_labels[key] = pred_label

    def compute_losses(self, outputs, labels, valid_masks, soft_labels):
        ce_loss = nn.CrossEntropyLoss()
        losses = []

        for sample_idx in range(outputs.size(0)):
            for annotator_idx in range(self.num_annotators):
                if valid_masks[sample_idx, annotator_idx]:
                    logits = outputs[sample_idx, annotator_idx]
                    target = labels[sample_idx, annotator_idx]
                    loss = ce_loss(logits.unsqueeze(0), target.unsqueeze(0))
                    losses.append(loss)
                elif soft_labels[sample_idx][annotator_idx] is not None:
                    logits = outputs[sample_idx, annotator_idx]
                    soft_target = soft_labels[sample_idx][annotator_idx]
                    pred_probs = F.softmax(logits, dim=0)
                    kl_loss = F.kl_div(
                        torch.log(pred_probs + 1e-8),
                        soft_target,
                        reduction='batchmean'
                    )
                    losses.append(kl_loss)

        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=self.device)


def train_model(backbone, dataloader_train, optimizer, scheduler, sim_trainer, num_annotators, args, epoch):
    backbone.train()
    device = next(backbone.parameters()).device
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader_train):
        optimizer.zero_grad()

        images, labels, valid_masks = batch
        images = images.cuda(args.local_rank, non_blocking=True)
        labels = labels.cuda(args.local_rank, non_blocking=True)
        valid_masks = valid_masks.cuda(args.local_rank, non_blocking=True)

        sim_trainer.similarity_manager.update_from_batch(labels, valid_masks)

        outputs, _ = backbone(images)

        soft_labels, confidence_scores = sim_trainer.generate_soft_labels(outputs.detach(), valid_masks)

        loss = sim_trainer.compute_losses(outputs, labels, valid_masks, soft_labels)

        if loss > 0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()
            scheduler.step(epoch, batch_idx)

        total_loss += loss.item()
        num_batches += 1

        sim_trainer.update_imputed_labels(batch_idx, soft_labels, confidence_scores)

        if batch_idx % 10 == 0:
            sim_trainer.similarity_manager.calculate_similarity_matrix()

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


def evaluate_model(backbone, dataloader_val, sim_trainer, num_annotators, args):
    backbone.eval()
    device = next(backbone.parameters()).device
    total_loss = 0.0
    num_batches = 0
    all_preds = [[] for _ in range(num_annotators)]
    all_labels = [[] for _ in range(num_annotators)]

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader_val):
            images, labels, valid_masks = batch
            images = images.cuda(args.local_rank, non_blocking=True)
            labels = labels.cuda(args.local_rank, non_blocking=True)
            valid_masks = valid_masks.cuda(args.local_rank, non_blocking=True)

            outputs, _ = backbone(images)

            soft_labels, _ = sim_trainer.generate_soft_labels(outputs, valid_masks)
            loss = sim_trainer.compute_losses(outputs, labels, valid_masks, soft_labels)

            total_loss += loss.item()
            num_batches += 1

            for sample_idx in range(outputs.size(0)):
                for annotator_idx in range(num_annotators):
                    if valid_masks[sample_idx, annotator_idx]:
                        pred = torch.argmax(outputs[sample_idx, annotator_idx]).item()
                        label = labels[sample_idx, annotator_idx].item()
                        all_preds[annotator_idx].append(pred)
                        all_labels[annotator_idx].append(label)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0

    accuracies = []
    maes = []
    for annotator_idx in range(num_annotators):
        if len(all_preds[annotator_idx]) > 0:
            preds = np.array(all_preds[annotator_idx])
            labels = np.array(all_labels[annotator_idx])
            accuracy = (preds == labels).mean()
            mae = np.abs(preds - labels).mean()
            accuracies.append(accuracy)
            maes.append(mae)
        else:
            accuracies.append(0.0)
            maes.append(0.0)

    return avg_loss, accuracies, maes


class LinearWarmupCosineLRScheduler:
    def __init__(self, optimizer, max_epoch, iters_per_epoch, min_lr, init_lr, warmup_steps=0, warmup_start_lr=-1):
        self.optimizer = optimizer
        self.max_epoch = max_epoch
        self.iters_per_epoch = iters_per_epoch
        self.min_lr = min_lr
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr
        self.total_steps = max_epoch * iters_per_epoch

    def step(self, cur_epoch, cur_step):
        total_cur_step = cur_epoch * self.iters_per_epoch + cur_step
        if total_cur_step < self.warmup_steps:
            self.warmup_lr_schedule(total_cur_step)
        else:
            self.cosine_lr_schedule(total_cur_step)

    def cosine_lr_schedule(self, step):
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        lr = (self.init_lr - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)) + self.min_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def warmup_lr_schedule(self, step):
        lr = min(self.init_lr,
                 self.warmup_start_lr + (self.init_lr - self.warmup_start_lr) * step / max(self.warmup_steps, 1))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(args):
    local_rank = args.local_rank
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    if local_rank == 0:
        wandb.init(project='SimLabel_STREET', name=f'SimLabel_{args.category}_miss{args.missing_rate}')

    train_dataset = ImageDataset(
        args.img_dir_train,
        args.csv_file_train,
        args.category,
        transform=data_transform,
        missing_rate=args.missing_rate
    )
    train_sampler = DistributedSampler(dataset=train_dataset)
    dataloader_train = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        sampler=train_sampler,
        num_workers=8
    )

    val_dataset = ImageDataset(
        args.img_dir_val,
        args.csv_file_val,
        args.category,
        transform=data_transform,
        missing_rate=args.missing_rate
    )
    val_sampler = DistributedSampler(dataset=val_dataset)
    dataloader_val = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        sampler=val_sampler,
        num_workers=8
    )

    if args.evaluate:
        test_dataset = ImageDataset(
            args.img_dir_test,
            args.csv_file_test,
            args.category,
            transform=data_transform,
            missing_rate=args.missing_rate
        )
        test_sampler = DistributedSampler(dataset=test_dataset)
        dataloader_test = DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            pin_memory=True,
            sampler=test_sampler,
            num_workers=8
        )

    NUM_ANNOTATORS = 10

    backbone = Backbone(
        vit_model=args.vit_model,
        q_former_model=args.q_former_model,
        attention_dim=256,
        num_levels=NUM_LEVELS,
        num_annotators=NUM_ANNOTATORS
    ).cuda(local_rank)

    backbone = DDP(backbone, device_ids=[local_rank], output_device=local_rank)

    total_steps = len(dataloader_train) * args.epochs
    warmup_steps = int(0.2 * total_steps)

    optimizer = AdamW(
        list(backbone.parameters()),
        lr=args.learning_rate,
        eps=1e-8,
        weight_decay=args.weight_decay
    )

    scheduler = LinearWarmupCosineLRScheduler(
        optimizer=optimizer,
        max_epoch=args.epochs,
        iters_per_epoch=len(dataloader_train),
        min_lr=1e-7,
        init_lr=args.learning_rate,
        warmup_steps=warmup_steps,
        warmup_start_lr=args.learning_rate / 10,
    )

    sim_trainer = SimLabelTrainer(
        num_annotators=NUM_ANNOTATORS,
        num_levels=NUM_LEVELS,
        confidence_threshold=args.confidence_threshold,
        device=f'cuda:{local_rank}'
    )

    if args.evaluate:
        checkpoint_path = os.path.join(args.save_path, args.checkpoint_name)
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{local_rank}')
        backbone.module.load_state_dict(checkpoint['model_state_dict'])

        test_loss, test_accuracies, test_maes = evaluate_model(backbone, dataloader_test, sim_trainer, NUM_ANNOTATORS,
                                                               args)

        if local_rank == 0:
            print(f"Test Loss: {test_loss:.4f}")
            avg_acc = sum(test_accuracies) / len(test_accuracies)
            avg_mae = sum(test_maes) / len(test_maes)
            print(f"Average Test Accuracy: {avg_acc:.4f}")
            print(f"Average Test MAE: {avg_mae:.4f}")

            for i, (acc, mae) in enumerate(zip(test_accuracies, test_maes)):
                print(f"Annotator {i + 1} - Accuracy: {acc:.4f}, MAE: {mae:.4f}")

            wandb.log({
                "test_loss": test_loss,
                "test_avg_accuracy": avg_acc,
                "test_avg_mae": avg_mae,
                **{f"test_acc_annotator_{i + 1}": acc for i, acc in enumerate(test_accuracies)},
                **{f"test_mae_annotator_{i + 1}": mae for i, mae in enumerate(test_maes)}
            })

        wandb.finish()
        return

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        train_loss = train_model(backbone, dataloader_train, optimizer, scheduler, sim_trainer, NUM_ANNOTATORS, args,
                                 epoch)
        val_loss, val_accuracies, val_maes = evaluate_model(backbone, dataloader_val, sim_trainer, NUM_ANNOTATORS, args)

        if local_rank == 0:
            print(f"Epoch {epoch + 1}/{args.epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            avg_acc = sum(val_accuracies) / len(val_accuracies)
            avg_mae = sum(val_maes) / len(val_maes)

            log_dict = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_avg_accuracy": avg_acc,
                "val_avg_mae": avg_mae
            }

            for i, (acc, mae) in enumerate(zip(val_accuracies, val_maes)):
                log_dict[f"val_acc_annotator_{i + 1}"] = acc
                log_dict[f"val_mae_annotator_{i + 1}"] = mae

            wandb.log(log_dict)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': backbone.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracies': val_accuracies,
                    'val_maes': val_maes
                }
                torch.save(checkpoint, os.path.join(args.save_path, 'best_model.pth'))

            if (epoch + 1) % args.save_epoch == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': backbone.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracies': val_accuracies,
                    'val_maes': val_maes
                }
                torch.save(checkpoint, os.path.join(args.save_path, f'checkpoint_epoch_{epoch + 1}.pth'))

    if local_rank == 0:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimLabel for STREET Dataset")
    parser.add_argument('--img_dir_train', type=str, default="data/street/images_train")
    parser.add_argument('--csv_file_train', type=str, default="data/street/train_data.csv")
    parser.add_argument('--img_dir_val', type=str, default="data/street/images_val")
    parser.add_argument('--csv_file_val', type=str, default="data/street/val_data.csv")
    parser.add_argument('--img_dir_test', type=str, default="data/street/images_test")
    parser.add_argument('--csv_file_test', type=str, default="data/street/test_data.csv")

    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--checkpoint_name', type=str, default='best_model.pth')

    parser.add_argument('--vit_model', type=str, default="checkpoints/eva_vit_g.pth")
    parser.add_argument('--q_former_model', type=str, default="checkpoints/instruct_blip_vicuna7b_trimmed.pth")

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--save_epoch', type=int, default=10)
    parser.add_argument('--save_path', type=str, default="checkpoints/simlabel_street")

    parser.add_argument('--category', type=str, default='Safe',
                        choices=['Happiness', 'Healthy', 'Safe', 'Lively', 'Orderly'])
    parser.add_argument('--missing_rate', type=float, default=0.4,
                        help='Simulated missing rate for STREET dataset (0.0-1.0)')
    parser.add_argument('--confidence_threshold', type=float, default=0.6)

    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True

    set_seed(42)

    main(args)
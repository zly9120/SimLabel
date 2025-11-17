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
import datetime
from torch.optim import AdamW
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
import random
from timechat.processors.video_processor import ToTHWC, ToUint8, load_video_with_resample
from timechat.processors import transforms_video, AlproVideoTrainProcessor
from timechat.models.blip2 import Blip2Base, disabled_train
from timechat.models.Qformer import BertConfig, BertLMHeadModel
import einops
import torch.nn.init as init

EMOTIONS = ['worried', 'happy', 'neutral', 'angry', 'surprise', 'sad', 'other', 'unknown']
transform = AlproVideoTrainProcessor(image_size=224).transform


class VideoDataset(Dataset):
    def __init__(self, video_dir, csv_file):
        self.video_dir = video_dir
        self.df = pd.read_csv(csv_file)
        self.num_annotators = sum(1 for col in self.df.columns if col.startswith('discrete'))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_name = f"{row['name']}.mp4"
        video_path = os.path.join(self.video_dir, video_name)

        video, _ = load_video_with_resample(
            video_path=video_path,
            n_frms=96,
            height=224,
            width=224,
            return_msg=True
        )
        video = transform(video)
        original_frames = video.size(1)

        emotions = [row[f'discrete{i + 1}'] for i in range(self.num_annotators)]
        emotion_labels = torch.zeros(self.num_annotators, len(EMOTIONS))
        valid_mask = torch.zeros(self.num_annotators, dtype=torch.bool)

        for i, emotion in enumerate(emotions):
            if pd.notna(emotion) and emotion in EMOTIONS:
                emotion_labels[i, EMOTIONS.index(emotion)] = 1
                valid_mask[i] = True

        return video, emotion_labels, valid_mask, original_frames


def dynamic_pad_collate_fn(batch):
    videos, emotion_labels, valid_masks, original_frames = zip(*batch)
    max_frames = max(video.size(1) for video in videos)

    padded_videos = []
    for video in videos:
        pad_size = max_frames - video.size(1)
        if pad_size > 0:
            padding = torch.zeros((video.size(0), pad_size, video.size(2), video.size(3)))
            padded_video = torch.cat([video, padding], dim=1)
        else:
            padded_video = video
        padded_videos.append(padded_video)

    padded_videos = torch.stack(padded_videos)
    emotion_labels = torch.stack(emotion_labels)
    valid_masks = torch.stack(valid_masks)
    original_frames = torch.tensor(original_frames)

    return padded_videos, emotion_labels, valid_masks, original_frames


class SimilarityManager:
    def __init__(self, num_annotators, device):
        self.num_annotators = num_annotators
        self.device = device
        self.similarity_matrix = None
        self.annotation_counts = torch.zeros((num_annotators, num_annotators), device=device)
        self.agreement_counts = torch.zeros((num_annotators, num_annotators), device=device)

    def update_from_batch(self, labels, valid_masks):
        batch_size = labels.size(0)

        for i in range(self.num_annotators):
            for j in range(i + 1, self.num_annotators):
                valid_both = valid_masks[:, i] & valid_masks[:, j]

                if valid_both.any():
                    labels_i = torch.argmax(labels[valid_both, i], dim=1)
                    labels_j = torch.argmax(labels[valid_both, j], dim=1)

                    agreements = (labels_i == labels_j).float().sum()
                    total = valid_both.float().sum()

                    self.agreement_counts[i, j] += agreements
                    self.agreement_counts[j, i] += agreements
                    self.annotation_counts[i, j] += total
                    self.annotation_counts[j, i] += total

    def calculate_similarity_matrix(self):
        self.similarity_matrix = torch.eye(self.num_annotators, device=self.device)

        for i in range(self.num_annotators):
            for j in range(i + 1, self.num_annotators):
                if self.annotation_counts[i, j] > 0:
                    n = self.annotation_counts[i, j]
                    po = self.agreement_counts[i, j] / n
                    pe = 1.0 / len(EMOTIONS)

                    if po != 1.0:
                        kappa = (po - pe) / (1.0 - pe)
                        kappa = max(0, kappa)
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
    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

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
            num_query_token=32,
            video_q_former_model="VL_LLaMA_2_7B_Finetuned.pth",
            max_frame_pos=96,
            frozen_video_Qformer=False,
            num_video_query_token=13,
            attention_dim=256,
            feature_dim=768,
            num_classes=8,
            num_annotators=13
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
        self.load_from_pretrained(multi_annotation=multi_annotation,
                                  url_or_filename=q_former_model)
        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False

        self.video_frame_position_embedding = nn.Embedding(max_frame_pos, self.Qformer.config.hidden_size)
        self.num_video_query_token = num_video_query_token
        self.video_Qformer, self.video_query_tokens = self.init_video_Qformer(
            num_query_token=num_video_query_token,
            vision_width=self.Qformer.config.hidden_size
        )

        self.video_Qformer.cls = None
        self.video_Qformer.bert.embeddings.word_embeddings = None
        self.video_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.video_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        if os.path.isfile(video_q_former_model):
            ckpt = torch.load(video_q_former_model, map_location="cpu")
            msg = self.load_state_dict(ckpt['model'], strict=False)

        if frozen_video_Qformer:
            for name, param in self.video_Qformer.named_parameters():
                param.requires_grad = False
            for name, param in self.video_frame_position_embedding.named_parameters():
                param.requires_grad = False
            self.video_query_tokens.requires_grad = False
        else:
            for name, param in self.video_Qformer.named_parameters():
                param.requires_grad = True
            for name, param in self.video_frame_position_embedding.named_parameters():
                param.requires_grad = True
            self.video_query_tokens.requires_grad = True

        self.vision_proj = nn.Sequential(
            nn.Linear(self.Qformer.config.hidden_size, attention_dim),
            nn.LayerNorm(attention_dim),
            nn.Dropout(0.3)
        )

        self.dropout = nn.Dropout(0.5)
        self.emotion_classifier = nn.ModuleList([
            nn.Sequential(
                nn.Linear(attention_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            ) for _ in range(num_annotators)
        ])

    def forward(self, unpadded_videos):
        video_hiddens = []
        video_cross_attentions = []

        for x in unpadded_videos:
            x = x.unsqueeze(0)
            batch_size, channel, time_length, height, width = x.size()
            device = x.device
            x = einops.rearrange(x, 'b c t h w -> (b t) c h w')

            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(x)).to(device)
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    output_attentions=False,
                    return_dict=True,
                )

                position_ids = torch.arange(time_length, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
                frame_position_embeddings = self.video_frame_position_embedding(position_ids)
                q_hidden_state = query_output.last_hidden_state

                frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
                frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h', b=batch_size,
                                                      t=time_length)
                frame_hidden_state = frame_position_embeddings + frame_hidden_state

                frame_hidden_state = einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h', b=batch_size,
                                                      t=time_length)
                frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)
                video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1)
                video_query_output = self.video_Qformer.bert(
                    query_embeds=video_query_tokens,
                    encoder_hidden_states=frame_hidden_state,
                    encoder_attention_mask=frame_atts,
                    output_attentions=True,
                    return_dict=True,
                )
                video_cross_attention = video_query_output.cross_attentions[10]
                video_hidden = video_query_output.last_hidden_state
                video_hidden = F.normalize(self.vision_proj(video_hidden), dim=-1)

            video_hiddens.append(video_hidden.squeeze(0))
            video_cross_attentions.append(video_cross_attention.squeeze(0))

        video_hiddens = torch.stack(video_hiddens)
        video_cross_attentions = torch.stack(video_cross_attentions)
        features = self.dropout(video_hiddens)

        outputs = []
        for annotator_idx in range(self.num_annotators):
            annotator_feature = features[:, annotator_idx, :]
            classifier = self.emotion_classifier[annotator_idx]
            outputs.append(classifier(annotator_feature))

        emotion_logits = torch.stack(outputs, dim=1)
        return emotion_logits, video_cross_attentions


class SimLabelTrainer:
    def __init__(self, num_annotators, confidence_threshold=0.6, device='cuda'):
        self.num_annotators = num_annotators
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.similarity_manager = SimilarityManager(num_annotators, device)
        self.imputed_labels = {}

    def generate_soft_labels(self, emotion_logits, valid_masks):
        batch_size = emotion_logits.size(0)
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
                    weighted_preds = torch.zeros(len(EMOTIONS), device=self.device)

                    for k in range(self.num_annotators):
                        if k != annotator_idx and valid_masks[sample_idx, k]:
                            pred_probs = F.softmax(emotion_logits[sample_idx, k], dim=0)
                            weighted_preds += weights[k] * pred_probs

                    if weighted_preds.sum() > 0:
                        weighted_preds = weighted_preds / weighted_preds.sum()

                        max_prob = weighted_preds.max()
                        entropy = -(weighted_preds * torch.log(weighted_preds + 1e-8)).sum()
                        max_entropy = math.log(len(EMOTIONS))
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

    def compute_losses(self, emotion_logits, emotion_labels, valid_masks, soft_labels):
        ce_loss = nn.CrossEntropyLoss()
        losses = []

        for sample_idx in range(emotion_logits.size(0)):
            for annotator_idx in range(self.num_annotators):
                if valid_masks[sample_idx, annotator_idx]:
                    logits = emotion_logits[sample_idx, annotator_idx]
                    target = torch.argmax(emotion_labels[sample_idx, annotator_idx])
                    loss = ce_loss(logits.unsqueeze(0), target.unsqueeze(0))
                    losses.append(loss)
                elif soft_labels[sample_idx][annotator_idx] is not None:
                    logits = emotion_logits[sample_idx, annotator_idx]
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

        padded_videos, emotion_labels, valid_masks, original_frames = batch
        padded_videos = padded_videos.cuda(args.local_rank, non_blocking=True)
        emotion_labels = emotion_labels.cuda(args.local_rank, non_blocking=True)
        valid_masks = valid_masks.cuda(args.local_rank, non_blocking=True)
        original_frames = original_frames.cuda(args.local_rank, non_blocking=True)

        sim_trainer.similarity_manager.update_from_batch(emotion_labels, valid_masks)

        unpadded_videos = []
        for i in range(len(padded_videos)):
            video = padded_videos[i]
            original_frame_count = original_frames[i].item()
            if original_frame_count < video.size(1):
                video = video[:, :original_frame_count]
            unpadded_videos.append(video)

        emotion_logits, _ = backbone(unpadded_videos)

        soft_labels, confidence_scores = sim_trainer.generate_soft_labels(emotion_logits.detach(), valid_masks)

        loss = sim_trainer.compute_losses(emotion_logits, emotion_labels, valid_masks, soft_labels)

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
            padded_videos, emotion_labels, valid_masks, original_frames = batch
            padded_videos = padded_videos.cuda(args.local_rank, non_blocking=True)
            emotion_labels = emotion_labels.cuda(args.local_rank, non_blocking=True)
            valid_masks = valid_masks.cuda(args.local_rank, non_blocking=True)
            original_frames = original_frames.cuda(args.local_rank, non_blocking=True)

            unpadded_videos = []
            for i in range(len(padded_videos)):
                video = padded_videos[i]
                original_frame_count = original_frames[i].item()
                if original_frame_count < video.size(1):
                    video = video[:, :original_frame_count]
                unpadded_videos.append(video)

            emotion_logits, _ = backbone(unpadded_videos)

            soft_labels, _ = sim_trainer.generate_soft_labels(emotion_logits, valid_masks)
            loss = sim_trainer.compute_losses(emotion_logits, emotion_labels, valid_masks, soft_labels)

            total_loss += loss.item()
            num_batches += 1

            for sample_idx in range(emotion_logits.size(0)):
                for annotator_idx in range(num_annotators):
                    if valid_masks[sample_idx, annotator_idx]:
                        pred = torch.argmax(emotion_logits[sample_idx, annotator_idx]).item()
                        label = torch.argmax(emotion_labels[sample_idx, annotator_idx]).item()
                        all_preds[annotator_idx].append(pred)
                        all_labels[annotator_idx].append(label)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0

    accuracies = []
    for annotator_idx in range(num_annotators):
        if len(all_preds[annotator_idx]) > 0:
            preds = np.array(all_preds[annotator_idx])
            labels = np.array(all_labels[annotator_idx])
            accuracy = (preds == labels).mean()
            accuracies.append(accuracy)
        else:
            accuracies.append(0.0)

    return avg_loss, accuracies


class WarmupCosineScheduler:
    def __init__(self, optimizer, max_epoch, iters_per_epoch, min_lr, init_lr, warmup_steps=0, warmup_start_lr=-1):
        self.optimizer = optimizer
        self.max_epoch = max_epoch
        self.iters_per_epoch = iters_per_epoch
        self.min_lr = min_lr
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else min_lr
        self.total_steps = max_epoch * iters_per_epoch

    def step(self, cur_epoch, cur_step):
        total_cur_step = cur_epoch * self.iters_per_epoch + cur_step

        if total_cur_step < self.warmup_steps:
            progress = float(total_cur_step) / float(max(1, self.warmup_steps))
            for group in self.optimizer.param_groups:
                lr = self.warmup_start_lr + progress * (group['lr'] - self.warmup_start_lr)
                group['lr'] = lr
        else:
            progress = float(total_cur_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            progress = min(1.0, max(0.0, progress))
            for group in self.optimizer.param_groups:
                lr = self.min_lr + 0.5 * (group['lr'] - self.min_lr) * (1.0 + math.cos(math.pi * progress))
                group['lr'] = lr


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
        wandb.init(project='SimLabel_AMER', name=f'SimLabel_confidence_{args.confidence_threshold}')

    train_dataset = VideoDataset(args.video_dir_train, args.csv_file_train)
    train_sampler = DistributedSampler(dataset=train_dataset)
    dataloader_train = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        sampler=train_sampler,
        num_workers=8,
        collate_fn=dynamic_pad_collate_fn
    )

    val_dataset = VideoDataset(args.video_dir_val, args.csv_file_val)
    val_sampler = DistributedSampler(dataset=val_dataset)
    dataloader_val = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        sampler=val_sampler,
        num_workers=8,
        collate_fn=dynamic_pad_collate_fn
    )

    if args.evaluate:
        test_dataset = VideoDataset(args.video_dir_test, args.csv_file_test)
        test_sampler = DistributedSampler(dataset=test_dataset)
        dataloader_test = DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            pin_memory=True,
            sampler=test_sampler,
            num_workers=8,
            collate_fn=dynamic_pad_collate_fn
        )

    NUM_CATEGORIES = len(EMOTIONS)
    NUM_ANNOTATORS = args.num_annotators

    backbone = Backbone(
        vit_model=args.vit_model,
        q_former_model=args.q_former_model,
        video_q_former_model=args.video_q_former_model,
        attention_dim=256,
        num_classes=NUM_CATEGORIES,
        num_annotators=NUM_ANNOTATORS
    ).cuda(local_rank)

    backbone = DDP(backbone, device_ids=[local_rank], output_device=local_rank)

    total_steps = len(dataloader_train) * args.epochs
    warmup_steps = int(0.2 * total_steps)

    optimizer = AdamW(
        backbone.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay
    )

    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        max_epoch=args.epochs,
        iters_per_epoch=len(dataloader_train),
        min_lr=1e-7,
        init_lr=args.learning_rate,
        warmup_steps=warmup_steps,
        warmup_start_lr=1e-7
    )

    sim_trainer = SimLabelTrainer(
        num_annotators=NUM_ANNOTATORS,
        confidence_threshold=args.confidence_threshold,
        device=f'cuda:{local_rank}'
    )

    if args.evaluate:
        checkpoint_path = os.path.join(args.save_path, args.checkpoint_name)
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{local_rank}')
        backbone.module.load_state_dict(checkpoint['model_state_dict'])

        test_loss, test_accuracies = evaluate_model(backbone, dataloader_test, sim_trainer, NUM_ANNOTATORS, args)

        if local_rank == 0:
            print(f"Test Loss: {test_loss:.4f}")
            for i, acc in enumerate(test_accuracies):
                print(f"Annotator {i + 1} Test Accuracy: {acc:.4f}")
            wandb.log({
                "test_loss": test_loss,
                **{f"test_acc_annotator_{i + 1}": acc for i, acc in enumerate(test_accuracies)}
            })

        wandb.finish()
        return

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        train_loss = train_model(backbone, dataloader_train, optimizer, scheduler, sim_trainer, NUM_ANNOTATORS, args,
                                 epoch)
        val_loss, val_accuracies = evaluate_model(backbone, dataloader_val, sim_trainer, NUM_ANNOTATORS, args)

        if local_rank == 0:
            print(f"Epoch {epoch + 1}/{args.epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            log_dict = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }

            for i, acc in enumerate(val_accuracies):
                print(f"Annotator {i + 1} Val Accuracy: {acc:.4f}")
                log_dict[f"val_acc_annotator_{i + 1}"] = acc

            wandb.log(log_dict)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': backbone.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracies': val_accuracies
                }
                torch.save(checkpoint, os.path.join(args.save_path, 'best_model.pth'))

            if (epoch + 1) % args.save_epoch == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': backbone.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracies': val_accuracies
                }
                torch.save(checkpoint, os.path.join(args.save_path, f'checkpoint_epoch_{epoch + 1}.pth'))

    if local_rank == 0:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimLabel for AMER Dataset")
    parser.add_argument('--video_dir_train', type=str, default="data/amer/video_train")
    parser.add_argument('--csv_file_train', type=str, default="data/amer/train_data.csv")
    parser.add_argument('--video_dir_val', type=str, default="data/amer/video_val")
    parser.add_argument('--csv_file_val', type=str, default="data/amer/val_data.csv")
    parser.add_argument('--video_dir_test', type=str, default="data/amer/video_test")
    parser.add_argument('--csv_file_test', type=str, default="data/amer/test_data.csv")

    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--checkpoint_name', type=str, default='best_model.pth')

    parser.add_argument('--vit_model', type=str, default="checkpoints/eva_vit_g.pth")
    parser.add_argument('--q_former_model', type=str, default="checkpoints/instruct_blip_vicuna7b_trimmed.pth")
    parser.add_argument('--video_q_former_model', type=str, default="checkpoints/VL_LLaMA_2_7B_Finetuned.pth")

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--save_epoch', type=int, default=10)
    parser.add_argument('--save_path', type=str, default="checkpoints/simlabel_amer")

    parser.add_argument('--num_annotators', type=int, default=13)
    parser.add_argument('--confidence_threshold', type=float, default=0.6)

    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True

    set_seed(42)

    main(args)
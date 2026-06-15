"""
Machine Translation: English to Bangla — Training Script with MLFlow
====================================================================

Fine-tunes a pretrained OPUS-MT model (shhossain/opus-mt-en-to-bn)
on a custom English→Bangla parallel corpus with full MLFlow experiment tracking.

Usage:
    python train.py --train_csv train.csv --val_csv val.csv --test_csv test.csv

MLFlow:
    Start the MLFlow UI with:  mlflow ui --port 5000
    The tracking URI defaults to a local ./mlruns directory.
"""

import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import Dataset, DataLoader
from torchmetrics.text import BLEUScore
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PRETRAINED_MODEL_NAME = "shhossain/opus-mt-en-to-bn"
MAX_SEQ_LENGTH = 128

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MTDataset(Dataset):
    """PyTorch Dataset that loads English→Bangla sentence pairs from a CSV."""

    def __init__(self, csv_file: str, tokenizer, max_length: int = MAX_SEQ_LENGTH):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        src_text = str(self.data.iloc[idx]["en"])
        tgt_text = str(self.data.iloc[idx]["bn"])

        src_encoding = self.tokenizer(
            src_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tgt_encoding = self.tokenizer(
            tgt_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "src_input_ids": src_encoding["input_ids"].squeeze(0),
            "src_attention_mask": src_encoding["attention_mask"].squeeze(0),
            "tgt_input_ids": tgt_encoding["input_ids"].squeeze(0),
            "tgt_attention_mask": tgt_encoding["attention_mask"].squeeze(0),
        }


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class MTDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for the MT datasets."""

    def __init__(
        self,
        train_csv: str,
        val_csv: str,
        test_csv: str,
        tokenizer,
        batch_size: int = 32,
        max_length: int = MAX_SEQ_LENGTH,
    ):
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            self.train_dataset = MTDataset(self.train_csv, self.tokenizer, self.max_length)
            self.val_dataset = MTDataset(self.val_csv, self.tokenizer, self.max_length)
        if stage == "test" or stage is None:
            self.test_dataset = MTDataset(self.test_csv, self.tokenizer, self.max_length)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------
class MTModel(pl.LightningModule):
    """PyTorch Lightning module wrapping the OPUS-MT seq2seq model."""

    def __init__(
        self,
        model_name: str = PRETRAINED_MODEL_NAME,
        learning_rate: float = 2e-5,
        scheduler_t_max: int = 10,
    ):
        super().__init__()
        # Save hyperparameters (excluding heavy tokenizer/model objects)
        self.save_hyperparameters(ignore=["model", "tokenizer"])

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.learning_rate = learning_rate
        self.scheduler_t_max = scheduler_t_max
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.bleu = BLEUScore()

    def forward(self, src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask):
        outputs = self.model(
            input_ids=src_input_ids,
            attention_mask=src_attention_mask,
            decoder_input_ids=tgt_input_ids[:, :-1],
            decoder_attention_mask=tgt_attention_mask[:, :-1],
        )
        return outputs

    def compute_loss(self, batch, stage: str):
        src_input_ids = batch["src_input_ids"]
        src_attention_mask = batch["src_attention_mask"]
        tgt_input_ids = batch["tgt_input_ids"]
        tgt_attention_mask = batch["tgt_attention_mask"]

        outputs = self(src_input_ids, src_attention_mask, tgt_input_ids, tgt_attention_mask)
        logits = outputs.logits

        loss = self.loss_fn(
            logits.view(-1, logits.size(-1)),
            tgt_input_ids[:, 1:].contiguous().view(-1),
        )

        # BLEU only on val / test (expensive to compute every training step)
        if stage in ("val", "test"):
            preds = torch.argmax(logits, dim=-1)
            pred_texts = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            tgt_texts = self.tokenizer.batch_decode(tgt_input_ids[:, 1:], skip_special_tokens=True)
            bleu_score = self.bleu(pred_texts, [[t] for t in tgt_texts])
            self.log(f"{stage}_bleu", bleu_score, prog_bar=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, "train")
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, "val")
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, "test")
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.scheduler_t_max
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# ---------------------------------------------------------------------------
# Translation helper (used outside training for demo / export)
# ---------------------------------------------------------------------------
def translate(model: MTModel, text: str, device: torch.device) -> str:
    """Run inference with the fine-tuned model on a single English sentence."""
    model.eval()
    model.to(device)

    encoding = model.tokenizer(
        text,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        generated_ids = model.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=MAX_SEQ_LENGTH,
            num_beams=4,
            early_stopping=True,
        )

    decoded = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return decoded


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train EN→BN machine translation model")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--val_csv", type=str, required=True, help="Path to validation CSV")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to test CSV")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--precision", type=str, default="16-mixed", help="Training precision")
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="mt_en_bn_finetune",
        help="MLFlow experiment name",
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        default=None,
        help="MLFlow tracking URI (default: local ./mlruns)",
    )
    parser.add_argument("--save_model_path", type=str, default="./saved_model",
                        help="Directory to save the final model and tokenizer")
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- MLFlow setup ----
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    mlflow_logger = MLFlowLogger(
        experiment_name=args.experiment_name,
        run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        tracking_uri=args.mlflow_tracking_uri,
        log_model=True,  # auto-log the LightningModule as an MLflow model
    )

    # Log all CLI hyperparameters
    mlflow_logger.log_hyperparams(vars(args))

    # ---- Tokenizer (shared) ----
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

    # ---- Data ----
    data_module = MTDataModule(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # ---- Model ----
    model = MTModel(
        model_name=PRETRAINED_MODEL_NAME,
        learning_rate=args.learning_rate,
        scheduler_t_max=args.max_epochs,
    )

    # ---- Callbacks ----
    checkpoint_callback = ModelCheckpoint(
        monitor="val_bleu",
        mode="max",
        save_top_k=2,
        filename="mt-en-bn-{epoch:02d}-{val_bleu:.3f}",
        save_weights_only=False,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # ---- Trainer ----
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=args.precision,
        log_every_n_steps=10,
        val_check_interval=0.25,
        logger=mlflow_logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
    )

    # ---- Train ----
    print("\n=== Starting training ===\n")
    trainer.fit(model, data_module)

    # ---- Test ----
    print("\n=== Running test ===\n")
    trainer.test(model, data_module)

    # ---- Save final model + tokenizer ----
    os.makedirs(args.save_model_path, exist_ok=True)
    model.model.save_pretrained(args.save_model_path)
    model.tokenizer.save_pretrained(args.save_model_path)
    print(f"\nModel and tokenizer saved to: {args.save_model_path}")

    # ---- Log saved model as MLFlow artifact ----
    mlflow_logger.experiment.log_artifact(
        run_id=mlflow_logger.run_id,
        local_path=args.save_model_path,
        artifact_path="final_model",
    )
    print("Model artifact logged to MLFlow.")

    # ---- Demo translation ----
    print("\n=== Demo translation ===")
    demo_text = "The man is walking in the park."
    translation = translate(model, demo_text, device)
    print(f"  EN: {demo_text}")
    print(f"  BN: {translation}")


if __name__ == "__main__":
    import mlflow
    main()

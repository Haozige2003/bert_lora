import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model

# 关掉烦人的警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SentimentTrainer:
    """情感分类模型训练器"""

    def __init__(self, model_name="bert-base-chinese"):
        self.model_name = model_name
        self.num_labels = 2
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")

    def load_data(self, csv_path):
        """读取 CSV 数据"""
        df = pd.read_csv(csv_path, encoding='utf-8')

        # 确保列名正确
        if 'label' not in df.columns or 'review' not in df.columns:
            if len(df.columns) >= 2:
                df.columns = ['label', 'review']
            else:
                raise ValueError("CSV 需要 'label' 和 'review' 两列")

        # 简单清洗
        df = df.dropna(subset=['review', 'label'])
        df['review'] = df['review'].astype(str).str.strip()
        df['label'] = df['label'].astype(int)
        df = df[df['review'].str.len() > 0]

        print(f"数据量: {len(df)}条 (正面:{sum(df['label'] == 1)}, 负面:{sum(df['label'] == 0)})")
        return df

    def split_data(self, df):
        """划分训练集和验证集"""
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        print(f"训练集: {len(train_df)}条 | 验证集: {len(val_df)}条")
        return train_df, val_df

    def tokenize(self, examples):
        """分词"""
        return self.tokenizer(
            examples['review'],
            truncation=True,
            padding=True,
            max_length=128
        )

    def prepare_dataset(self, train_df, val_df):
        """准备数据集"""
        from datasets import Dataset

        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)

        train_dataset = train_dataset.rename_column("label", "labels")
        val_dataset = val_dataset.rename_column("label", "labels")

        train_dataset = train_dataset.map(self.tokenize, batched=True)
        val_dataset = val_dataset.map(self.tokenize, batched=True)

        train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        return train_dataset, val_dataset

    def build_model(self):
        """构建带 LoRA 的模型"""
        print("加载模型...")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)

        base_model = BertForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )

        # LoRA 配置
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["query", "key", "value", "output.dense"],
            lora_dropout=0.05,
            bias="lora_only",
            modules_to_save=["classifier"]
        )

        self.model = get_peft_model(base_model, lora_config)
        self.model.to(self.device)

        # 打印参数量
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"总参数: {total:,} | 可训练: {trainable:,} ({trainable / total:.2%})")

    def compute_metrics(self, pred):
        """计算准确率"""
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted")
        }

    def train(self, train_dataset, val_dataset, output_dir):
        """开始训练"""
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=4,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=3e-4,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            fp16=(self.device == "cuda"),
            logging_steps=10,
            save_total_limit=1,  # 只保留最新模型
            max_grad_norm=1.0,
            warmup_ratio=0.1,
            gradient_accumulation_steps=2
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            compute_metrics=self.compute_metrics
        )

        print("开始训练...")
        trainer.train()
        trainer.save_model(output_dir)
        print(f"模型保存到: {output_dir}")


def main():
    # 配置
    MODE = "train"  # train: 训练, predict: 预测（这里只保留训练）
    TRAIN_FILE = "shop_2class.csv"
    MODEL_DIR = "./shop_sentiment_model"

    trainer = SentimentTrainer()

    if MODE == "train":
        print("=== 训练模式 ===")
        df = trainer.load_data(TRAIN_FILE)
        train_df, val_df = trainer.split_data(df)
        trainer.build_model()
        train_dataset, val_dataset = trainer.prepare_dataset(train_df, val_df)
        trainer.train(train_dataset, val_dataset, MODEL_DIR)
        print("训练完成！")


if __name__ == "__main__":
    main()
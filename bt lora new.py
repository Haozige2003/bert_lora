import torch
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import os
import json


class ShopSentimentTrainer:
    """基于 BERT 与 LoRA 的情感分类微调 Pipeline"""

    def __init__(self, model_name="bert-base-chinese", num_labels=2):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = None
        self.model = None

    def load_data(self, csv_path, text_col="review", label_col="label"):
        """加载并清洗数据"""
        df = pd.read_csv(csv_path, encoding='utf-8')

        assert text_col in df.columns, f"缺少列: {text_col}"
        assert label_col in df.columns, f"缺少列: {label_col}"

        df = df.dropna(subset=[text_col, label_col])
        df[text_col] = df[text_col].astype(str).str.strip()
        df = df[df[text_col].str.len() > 0]

        print(f"数据加载完成: {len(df)} 条样本 (正面: {sum(df[label_col] == 1)}, 负面: {sum(df[label_col] == 0)})")
        return df

    def prepare_datasets(self, df, test_size=0.2, val_size=0.1):
        """划分训练、验证与测试集并转换为 HuggingFace Dataset"""
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, stratify=df["label"], random_state=42
        )

        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size / (1 - test_size),
            stratify=train_val_df["label"], random_state=42
        )

        datasets = DatasetDict({
            "train": Dataset.from_pandas(train_df[["label", "review"]]),
            "validation": Dataset.from_pandas(val_df[["label", "review"]]),
            "test": Dataset.from_pandas(test_df[["label", "review"]])
        })

        return datasets.rename_columns({"review": "text", "label": "labels"})

    def setup_lora(self, r=16, alpha=32):
        """注入 LoRA 适配器并统计参数量"""
        base_params = sum(p.numel() for p in self.model.parameters())

        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["query", "key", "value", "output.dense"],
            lora_dropout=0.05,
            bias="lora_only",
            task_type="SEQ_CLS",
            modules_to_save=["classifier"]
        )

        self.model = get_peft_model(self.model, lora_config)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"LoRA 配置: rank={r}, alpha={alpha}")
        print(f"可训练参数量: {trainable_params:,} / {base_params:,} ({trainable_params / base_params:.2%})")

        return lora_config

    def tokenize_data(self, datasets, max_length=128):
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=max_length
            )

        tokenized = datasets.map(tokenize_function, batched=True)
        tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        return tokenized

    def train(self, train_dataset, val_dataset, output_dir="./model_output"):
        def compute_metrics(pred):
            predictions = np.argmax(pred.predictions, axis=1)
            labels = pred.label_ids
            return {
                "accuracy": accuracy_score(labels, predictions),
                "f1": f1_score(labels, predictions, average="weighted")
            }

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=4,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=16,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-4,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            fp16=True,
            max_grad_norm=1.0,
            warmup_ratio=0.1,
            dataloader_num_workers=0,
            gradient_accumulation_steps=2
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=compute_metrics
        )

        print("开始训练...")
        train_result = trainer.train()
        trainer.save_model(f"{output_dir}/final_model")

        return trainer, train_result

    def evaluate(self, trainer, test_dataset):
        print("评估模型中...")
        eval_result = trainer.evaluate(test_dataset)
        print(f"测试集准确率: {eval_result['eval_accuracy']:.2%} | F1分数: {eval_result['eval_f1']:.4f}")
        return eval_result

    def predict_examples(self, texts):
        if self.tokenizer is None or self.model is None:
            raise ValueError("请先初始化模型和分词器后再进行预测。")

        self.model.eval()
        results = []
        device = next(self.model.parameters()).device

        for text in texts:
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
                pred = torch.argmax(probs).item()
                confidence = probs[pred].item()

            results.append({
                "text": text[:50] + ("..." if len(text) > 50 else ""),
                "sentiment": "正面" if pred == 1 else "负面",
                "confidence": confidence
            })

        return results

    def run_pipeline(self, csv_path, output_dir="./shop_sentiment_model"):
        df = self.load_data(csv_path)
        datasets = self.prepare_datasets(df)

        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )

        self.setup_lora(r=16, alpha=32)
        tokenized = self.tokenize_data(datasets)

        trainer, _ = self.train(tokenized["train"], tokenized["validation"], output_dir)
        eval_result = self.evaluate(trainer, tokenized["test"])

        training_info = {
            "dataset": os.path.basename(csv_path),
            "samples": len(df),
            "accuracy": float(eval_result["eval_accuracy"]),
            "f1": float(eval_result["eval_f1"]),
            "model": self.model_name,
            "technique": "LoRA 微调"
        }

        with open(os.path.join(output_dir, "training_info.json"), 'w', encoding='utf-8') as f:
            json.dump(training_info, f, indent=2, ensure_ascii=False)

        print(f"流程完成。模型已保存至: {output_dir}")
        return training_info


def main():
    csv_file = "shop_2class.csv"
    if not os.path.exists(csv_file):
        print(f"错误: 数据集文件 {csv_file} 未找到。")
        return

    if torch.cuda.is_available():
        device_info = torch.cuda.get_device_name(0)
        print(f"使用设备: {device_info}")
    else:
        print("使用设备: CPU")

    trainer = ShopSentimentTrainer(model_name="bert-base-chinese")
    results = trainer.run_pipeline(csv_file)

    print("\n--- 示例预测 ---")
    test_texts = [
        "灯光角度固定，无法根据需求调整！",
        "工作人员拍照手法生硬，不会引导姿势",
        "租赁服装款式新颖多样，面料舒适干净无异味！"
    ]

    predictions = trainer.predict_examples(test_texts)
    for pred in predictions:
        print(f"文本: {pred['text']}\n  -> 情感: {pred['sentiment']} (置信度: {pred['confidence']:.1%})")


if __name__ == "__main__":
    main()
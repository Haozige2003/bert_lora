import pandas as pd
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelTester:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.id2label = {0: "负面", 1: "正面"}

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(device)
        self.model.eval()

        if hasattr(self.model.config, 'id2label'):
            self.id2label = self.model.config.id2label

    def test(self, test_file):
        df = pd.read_csv(test_file)

        if 'label' not in df.columns or 'review' not in df.columns:
            if len(df.columns) >= 2:
                df.columns = ['label', 'review']
            else:
                raise ValueError("CSV 需要 'label' 和 'review' 列")

        texts = df['review'].astype(str).tolist()
        true_labels = df['label'].astype(int).tolist()

        all_preds = []
        batch_size = 16

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, truncation=True, padding=True, max_length=128, return_tensors='pt').to(
                device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())

        # 计算指标
        acc = accuracy_score(true_labels, all_preds)
        f1 = f1_score(true_labels, all_preds, average='weighted')

        print(f"准确率: {acc:.2%}")
        print(f"F1分数: {f1:.4f}")
        print(f"总样本数: {len(texts)}")

        # 找出预测错误的样本
        wrong_samples = []
        for idx, (text, true, pred) in enumerate(zip(texts, true_labels, all_preds)):
            if true != pred:
                wrong_samples.append({
                    "index": idx,
                    "text": text[:100],
                    "true_label": true,
                    "pred_label": pred
                })

        print(f"\n预测错误的样本数: {len(wrong_samples)}")
        for sample in wrong_samples[:5]:  # 只打印前5个
            print(f"  样本 {sample['index']}: 真实={sample['true_label']}, 预测={sample['pred_label']}")
            print(f"    文本: {sample['text']}...")

        # 保存结果
        results = []
        for text, true, pred in zip(texts, true_labels, all_preds):
            results.append({
                "text": text,
                "true_label": int(true),
                "pred_label": int(pred),
                "is_correct": bool(true == pred)
            })

        with open("test_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"结果已保存到 test_results.json")


def main():
    TEST_FILE = "test_reviews.csv"
    MODEL_DIR = "./shop_sentiment_model"

    tester = ModelTester(MODEL_DIR)
    tester.load_model()
    tester.test(TEST_FILE)


if __name__ == "__main__":
    main()
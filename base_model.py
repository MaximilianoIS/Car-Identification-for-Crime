import os
import json
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import wandb

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from timm import create_model


# ✅ 사용자 정의 Dataset 클래스
class BrandCSVImageDataset(Dataset):
    def __init__(self, df, img_dir, label_encoder, model_encoder, transform=None):
        self.data = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.label_encoder = label_encoder
        self.model_encoder = model_encoder
        self.transform = transform

        # brand → 숫자 label
        self.data['brand_label'] = self.label_encoder.transform(self.data['brand'])
        self.data['model_label'] = self.model_encoder.transform(self.data['model'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, row['brand_label'], row['model_label']


class BrandThenModelClassifier(nn.Module):
    def __init__(self, num_brands, num_models):
        super().__init__()
        self.backbone = create_model("convnext_base", pretrained=True, num_classes=0, global_pool="avg")
        self.feature_dim = self.backbone.num_features
        self.brand_classifier = nn.Linear(self.feature_dim, num_brands)
        self.model_classifier = nn.Linear(self.feature_dim + num_brands, num_models)

    def forward(self, x):
        feat = self.backbone(x)  # [B, feature_dim]
        brand_logits = self.brand_classifier(feat)  # [B, num_brands]
        brand_onehot = torch.nn.functional.one_hot(brand_logits.argmax(dim=1), num_classes=brand_logits.shape[1]).float()
        x_cat = torch.cat([feat, brand_onehot], dim=1)  # [B, feature_dim + num_brands]
        model_logits = self.model_classifier(x_cat)
        return brand_logits, model_logits


# ✅ 메인 실행 코드
def main():
    # 경로 설정
    csv_path = "dataset/augmented_split_labeled.csv"
    img_dir = "dataset/ㅋ"

    # CSV 로드 및 클래스 인코딩
    df = pd.read_csv(csv_path)
    # filename, brand, model 중 하나라도 NaN인 샘플 제거
    df = df.dropna(subset=["filename", "brand", "model"])
    # 'unknown' 모델 제거
    df = df[df['model'] != 'unknown']

    # 2. 브랜드별 개수 확인
    brand_counts = df['brand'].value_counts()

    # 3. 문제 있는 브랜드 출력
    problematic = brand_counts[brand_counts < 2]
    if not problematic.empty:
        print("❗ Stratify 불가능한 브랜드 (샘플 수 < 2):")
        print(problematic)

    # 4. 문제 있는 브랜드 제거
    valid_brands = brand_counts[brand_counts >= 2].index
    df_filtered = df[df['brand'].isin(valid_brands)].copy().reset_index(drop=True)

    # 클래스 목록 저장 및 label encoder 적용
    label_encoder = LabelEncoder()
    # 저장: brand_classes.txt
    with open("brand_classes.txt", "w") as f:
        for cls in sorted(df_filtered['brand'].unique()):
            f.write(f"{cls}\n")
    # 불러오기 및 encoder 적용 (학습/추론 일관성)
    with open("brand_classes.txt", "r") as f:
        class_list = [line.strip() for line in f]
    label_encoder.fit(class_list)
    num_classes = len(label_encoder.classes_)
    print(f"✅ 클래스 수: {num_classes} → {list(label_encoder.classes_)}")

    # 모델 클래스 인코딩
    model_encoder = LabelEncoder()
    df_model = df_filtered.dropna(subset=["model"])
    with open("model_classes.txt", "w") as f:
        for cls in sorted(df_model["model"].unique()):
            f.write(f"{cls}\n")
    with open("model_classes.txt", "r") as f:
        model_class_list = [line.strip() for line in f]
    model_encoder.fit(model_class_list)
    num_model_classes = len(model_encoder.classes_)
    print(f"✅ 모델 클래스 수: {num_model_classes}")

    # 브랜드별 모델 매핑 저장
    brand_model_map = {}
    for brand in df_filtered['brand'].unique():
        brand_models = df_filtered[df_filtered['brand'] == brand]['model'].dropna().unique().tolist()
        brand_model_map[brand] = brand_models
    with open("brand_model_mapping.json", "w") as f:
        json.dump(brand_model_map, f, indent=2)
    print("✅ 브랜드별 모델 매핑 저장 완료: brand_model_mapping.json")

    # 'split' 컬럼 기준으로 train/val 분리
    df_train = df_filtered[df_filtered["split"] == "train"].copy()
    df_val = df_filtered[df_filtered["split"] == "val"].copy()

    # 이미지 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Dataset & DataLoader
    train_dataset = BrandCSVImageDataset(df_train, img_dir, label_encoder, model_encoder, transform)
    val_dataset = BrandCSVImageDataset(df_val, img_dir, label_encoder, model_encoder, transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    # 모델 정의 (BrandThenModelClassifier)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BrandThenModelClassifier(num_brands=num_classes, num_models=num_model_classes).to(device)
    import os
    os.environ["WANDB_MODE"] = "offline"
    # wandb init
    wandb.init(project="brand_model_classifier", name="convnext_base_run")

    # 손실함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    start_epoch = 1
    if os.path.exists("brand_classifier_checkpoint.pth"):
        checkpoint = torch.load("brand_classifier_checkpoint.pth", map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint.get('epoch', 1) + 1
        print("✅ 체크포인트에서 모델과 옵티마이저 로드 완료")

    end_epoch = 6

    best_val_acc = 0
    best_model_path = None

    metrics_log = []

    # 학습 루프
    for epoch in range(start_epoch, end_epoch):
        model.train()
        train_loss = 0
        for batch_idx, (imgs, brand_labels, model_labels) in enumerate(train_loader):
            print(f"Epoch {epoch}, Batch {batch_idx+1}/{len(train_loader)}")
            imgs = imgs.to(device)
            brand_labels = brand_labels.to(device)
            model_labels = model_labels.to(device)
            brand_logits, model_logits = model(imgs)
            loss_brand = criterion(brand_logits, brand_labels)
            loss_model = criterion(model_logits, model_labels)
            loss = loss_model + loss_brand

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 검증
        model.eval()
        brand_correct, brand_total = 0, 0
        model_correct, model_total = 0, 0
        val_loss = 0
        with torch.no_grad():
            for imgs, brand_labels, model_labels in val_loader:
                imgs = imgs.to(device)
                brand_labels = brand_labels.to(device)
                model_labels = model_labels.to(device)
                brand_logits, model_logits = model(imgs)

                loss_brand = criterion(brand_logits, brand_labels)
                loss_model = criterion(model_logits, model_labels)
                loss = loss_model + loss_brand
                val_loss += loss.item()

                # 브랜드 정확도 (기존과 동일)
                brand_preds = brand_logits.argmax(1)
                brand_correct += (brand_preds == brand_labels).sum().item()
                brand_total += brand_labels.size(0)

                # 모델 정확도 (브랜드 기반 제한 적용)
                for i in range(imgs.size(0)):
                    pred_brand_idx = brand_preds[i].item()
                    brand_name = label_encoder.classes_[pred_brand_idx]

                    # 허용된 모델 목록
                    allowed_models = brand_model_map.get(brand_name, [])
                    if not allowed_models:
                        continue
                    allowed_model_indices = [model_encoder.transform([m])[0] for m in allowed_models if m in model_encoder.classes_]

                    if not allowed_model_indices:
                        continue

                    logits_i = model_logits[i][allowed_model_indices]
                    pred_model_local_idx = logits_i.argmax().item()
                    pred_model = allowed_models[pred_model_local_idx]

                    true_model = model_encoder.inverse_transform([model_labels[i].item()])[0]
                    if pred_model == true_model:
                        model_correct += 1
                    model_total += 1
        brand_acc = brand_correct / brand_total * 100
        model_acc = model_correct / model_total * 100 if model_total > 0 else 0.0
        avg_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Brand Val Acc: {brand_acc:.2f}% | Model Val Acc: {model_acc:.2f}%")

        metrics_log.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_loss": avg_val_loss,
            "brand_val_acc": brand_acc,
            "model_val_acc": model_acc
        })
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_loss": avg_val_loss,
            "brand_val_acc": brand_acc,
            "model_val_acc": model_acc
        })

        # 모델 분류기 기준 best_val_acc 사용
        if model_acc > best_val_acc:
            best_val_acc = model_acc
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
            best_model_path = f"brand_classifier_best_{epoch}.pth"
            torch.save(model, best_model_path)
            print(f"🌟 최고 성능 갱신! 모델 저장됨: epoch {epoch}, Model Val Acc: {model_acc:.2f}%")

        scheduler.step(model_acc)


    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch
    }, "brand_classifier_checkpoint.pth")
    print(f"✅ 체크포인트 저장 완료: epoch {epoch}")
    torch.save(model, "brand_classifier_full.pth")
    print("✅ 모델 저장 완료: brand_classifier.pth")

    pd.DataFrame(metrics_log).to_csv("training_log.csv", index=False)
    print("✅ 학습 로그 저장 완료: training_log.csv")
    wandb.save("training_log.csv")
    if best_model_path is not None:
        wandb.save(best_model_path)


if __name__ == "__main__":
    main()
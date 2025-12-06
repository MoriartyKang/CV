from transformers import AutoImageProcessor, ViTForImageClassification
import torch
from PIL import Image

# 3개의 이미지 파일을 열어 PIL Image 객체 리스트로 저장
img=[Image.open('251125\data\BSDS_242078.jpg'),Image.open('251125\data\BSDS_361010.jpg'),Image.open('251125\data\BSDS_376001.jpg')]

# 사전 학습된 ViT 모델 불러오기 (전처리, 가중치)
model_name = 'google/vit-base-patch16-224'
image_processor = AutoImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# 이미지 전처리 후 결과 텐서를 torch 형태로 반환
inputs = image_processor(images=img, return_tensors='pt')
res = model(**inputs) # softmax 적용 전의 값을 res.logits에 저장함

import matplotlib.pyplot as plt

for i in range(res.logits.shape[0]):
    plt.imshow(img[i]); plt.xticks([]); plt.yticks([]); plt.show()

    predicted_label_tensor = torch.argmax(res.logits[i], dim=-1) # # 해당 이미지의 로짓(res.logits[i]) 중 가장 높은 값의 인덱스 사용
    predicted_label = predicted_label_tensor.item()

    probabilities = torch.softmax(res.logits[i], dim=0) # 로짓에 소프트맥스 사용해서 확률 예측
    prob = probabilities[predicted_label].item() * 100.0

    print(i,'번째 영상의 1순위 부류: ', model.config.id2label[predicted_label],prob)
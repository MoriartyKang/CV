from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# 1개의 이미지 파일을 열어 PIL Image 객체 리스트로 저장
img=Image.open('251125\data\BSDS_361010.jpg')

# 사전 학습된 CLIP 모델 불러오기 (전처리, 가중치)
model_name = 'openai/clip-vit-base-patch32'
processor=CLIPProcessor.from_pretrained(model_name)
model=CLIPModel.from_pretrained(model_name)

# 영상과 비교할 입력 텍스트 4개 정의
captions=['Two horses are running on grass', 'Students are eating', 'Croquet playing on horses', 'Golf playing on horses']
inputs=processor(text=captions,images=img,return_tensors='pt',padding=True)
res=model(**inputs) # res에는 이미지와 텍스트 쌍의 유사도를 계산

import matplotlib.pyplot as plt
plt.imshow(img); plt.xticks([]); plt.yticks([]); plt.show()

# 각 텍스트가 이미지에 적합할 확률 계산 및 출력
logits=res.logits_per_image
probs=logits.softmax(dim=1)
for i in range(len(captions)):
    print(captions[i],': ','{:.2f}'.format(float(probs[0,i].detach() * 100.0)))
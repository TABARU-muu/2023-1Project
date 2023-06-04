
import torch
import clip
from PIL import Image

# 음식 카테고리 리스트
food_categories = [
    "chicken", "Grilled pork", "chiness food", "sea food", "noodle",
    "Grilled fish", "Grilled Cutlassfish", "beef", "Hanjeongsik", "sashimi",
    "bolied pork", "hamburger", "shrimp sashimi", "shrimp", "melon prosciutto",
    "pork cutlet", "bread", "pizza", "Waffle", "Tteokbokki",
    "pasta", "ramen", "sushi", "corn dog", "American breakfast",
    "crab", "curry", "bread", "soup", "tuna",
    "doughnut", "koreanstyle sushi(gimbab)"
]

# CLIP 모델 및 토크나이저 불러오기
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 이미지를 카테고리로 라벨링하는 함수
def label_food_category(image_path):
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        text_features = model.encode_text(clip.tokenize(food_categories).to(device))
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        predicted_category = food_categories[similarity.argmax()]

    return predicted_category

# 테스트 이미지 경로
image_path = "test_image.jpg"

# 음식 카테고리 라벨링 실행
predicted_category = label_food_category(image_path)
print("Predicted Category:", predicted_category)

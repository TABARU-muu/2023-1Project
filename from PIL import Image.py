from PIL import Image

def recommend_travel_course(photo_path):
    # 사진 로드
    image = Image.open(photo_path)
    image.show()

    # 여행 코스 추천 로직
    # ...

# 여행 코스 추천 시스템 호출
photo_path = 'C:\Users\namik\Desktop\Recomendataion_Place\Recommendataion_Place\tourist_images'  # 사용할 사진 파일 경로
recommend_travel_course(photo_path)

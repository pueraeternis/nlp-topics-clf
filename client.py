import requests

text = """
    Зимой я люблю кататься на лыжах. Как правило, я просто бегаю по парку.
    Но иногда я езжу в Красную Поляну, где катаюсь на горных лыжах всю зиму.
    Благо, у меня удаленная работа, и я могу совмещать любимый спорт, работу
    и страсть к путешествиям.
"""

url = "http://localhost:8080/predict"

data = {"text": text}

response = requests.post(url, json=data)

if response.status_code == 200:
    prediction = response.json()
    print(f"Предсказанный класс: {prediction['label']}")
    print("Вероятности для классов:")
    for label, prob in prediction["probabilities"].items():
        print(f"{label}: {prob}")
else:
    print(f"Ошибка: {response.status_code}, {response.text}")

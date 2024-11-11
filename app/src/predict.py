from dataclasses import dataclass, field

from sklearn.pipeline import Pipeline

from app.src.processing import process_text


@dataclass
class Prediction:
    label: str
    probabilities: dict[str, float] = field(default_factory=dict)


def predict(text: str, model: Pipeline) -> Prediction:
    text_processed = process_text(text)

    probs = model.predict_proba([text_processed])[0].round(4).tolist()
    class_names = model.classes_
    probabilities = {class_name: prob for class_name, prob in zip(class_names, probs)}
    label = max(probabilities, key=lambda k: probabilities[k])

    return Prediction(label=label, probabilities=probabilities)

import numpy as np
from sklearn.base import defaultdict
from Utilities.feedback_utilities import BASE_TEMPLATE_EN, EMOTION_LABELS_EN


class FeedbackCoach:
    def aggregate_stream_scores(time_slots: list, stream_name: str) -> dict[str, float]:
        emotion_values = defaultdict(list)

        for ts in time_slots:
            stream_data = ts["modal"].get(stream_name, {})
            emotions = stream_data.get("emotions", {})

            if not emotions:
                continue

            for emotion, score in emotions.items():
                emotion_values[emotion].append(score)

        if not emotion_values:
            return {}

        return {
            emotion: float(np.mean(scores))
            for emotion, scores in emotion_values.items()
        }

    def analyze_score_vector(self, score_vector: dict[str, float]):
        if not score_vector:
            return None

        sorted_emotions = sorted(
            score_vector.items(),
            key=lambda x: x[1],
            reverse=True
        )

        primary_emotion, primary_score = sorted_emotions[0]

        secondary_emotion = None
        secondary_score = None

        if len(sorted_emotions) > 1:
            secondary_emotion, secondary_score = sorted_emotions[1]

        return {
            "primary_emotion": primary_emotion,
            "primary_score": primary_score,
            "secondary_emotion": secondary_emotion,
            "secondary_score": secondary_score
        }

    def score_to_intensity(self, score: float) -> str:
        if score < 0.2:
            return "subtle"
        elif score < 0.5:
            return "moderate"
        elif score < 0.8:
            return "strong"
        else:
            return "very strong"

    def generate_stream_feedback(
        self,
        score_vector: dict[str, float],
        stream_name: str
    ) -> str:

        analysis = self.analyze_score_vector(score_vector)

        if analysis is None:
            return f"[{stream_name.upper()}] Stream not available or no emotional information detected."

        primary_emotion = EMOTION_LABELS_EN.get(
            analysis["primary_emotion"],
            analysis["primary_emotion"]
        )

        primary_intensity = self.score_to_intensity(
            analysis["primary_score"]
        )

        if analysis["secondary_emotion"] is not None:
            secondary_emotion = EMOTION_LABELS_EN.get(
                analysis["secondary_emotion"],
                analysis["secondary_emotion"]
            )

            text = BASE_TEMPLATE_EN.format(
                primary_emotion=primary_emotion,
                primary_intensity=primary_intensity,
                secondary_emotion=secondary_emotion
            )
        else:
            text = (
                f"The emotional state is dominated by {primary_emotion} "
                f"with a {primary_intensity} intensity."
            )

        return f"[{stream_name.upper()}] {text}"

    def generate_full_feedback(
        self,
        stream_scores: dict[str, dict[str, float]]
    ) -> str:

        feedback_parts = []

        for stream_name, scores in stream_scores.items():
            feedback_parts.append(
                self.generate_stream_feedback(scores, stream_name)
            )

        return "\n".join(feedback_parts)
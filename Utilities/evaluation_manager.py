import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from Utilities.utils import Utils

class EvaluationManager:
    def __init__(self, utils: Utils):
        self.utils = utils

    def _emotion_dict_to_label(self, emotions: dict) -> int:
        """
        Converte il dizionario delle emozioni nella label predetta
        usando argmax.
        """
        predict_label = max(emotions.items(), key=lambda x: x[1])[0]
        
        return self.utils.encode_ekman_labels(predict_label)

    def fuse_emotions_average(self, modal_emotions: dict) -> dict:
        """
        modal_emotions = {
            "video": {"anger": 0.1, ...},
            "audio": {"anger": 0.2, ...},
            "text":  {"anger": 0.3, ...}
        }

        Ritorna:
            {"anger": 0.2, ...}
        """
        fused = {}
        count = 0

        for emotions in modal_emotions.values():
            if not emotions:
                continue

            for emo, score in emotions.items():
                fused[emo] = fused.get(emo, 0.0) + score

            count += 1

        if count == 0:
            return {}

        for emo in fused:
            fused[emo] /= count

        return fused


    def evaluate_stream(self, stream_name: str, video_name_list: list) -> dict:
        logger = self.utils.setup_logger()
        logger.info(f"Evaluating stream: {stream_name}")

        result_json = pd.read_json(
            self.utils.config["Paths"]["complete_info_path"]
        )
         
        # Escludo il file json se presente
        video_name_list = [n for n in video_name_list if "json" not in n]

        video_metrics = []

        # =============================
        # LOOP PER VIDEO
        # =============================
        for video in video_name_list:
            video_name = video.split("/")[1]

            if video_name not in result_json:
                logger.warning(f"Video {video_name} non trovato nei risultati.")
                continue

            video_data = result_json[video_name]["time_slots"]

            y_true = []
            y_pred = []

            # prima del loop time_slot:
            emotion_order = self.utils.config.get("EmotionOrder", None)
            # se non hai una lista in config, la puoi derivare dalla prima occorrenza non vuota
            prob_seq = []
            conf_seq = []

            for time_slot in video_data:
                gt = time_slot["ground_truth"]

                modal_data = time_slot["modal"].get(stream_name, {})
                emotions = modal_data.get("emotions", {})

                if not emotions:
                    continue

                # Se non hai emotion_order, inizializzalo dalla prima chiave vista (meglio comunque fissarlo in config!)
                if emotion_order is None:
                    emotion_order = sorted(list(emotions.keys()))

                pred = self._emotion_dict_to_label(emotions)

                # >>> aggiunta per TCS
                p_t = self._emotion_dict_to_vector(emotions, emotion_order)
                if p_t.sum() > 0:
                    prob_seq.append(p_t)
                    conf_seq.append(float(p_t.max()))
                # <<<

                y_true.append(gt)
                y_pred.append(pred)


            if not y_true:
                logger.warning(
                    f"Nessun time-slot valido per {stream_name} nel video {video_name}"
                )
                continue

            # =============================
            # METRICHE PER VIDEO
            # =============================
            tcs = self._temporal_consistency_score(prob_seq, weight_seq=conf_seq)
            metrics = {
                "video": video_name,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
                "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
                "f1_score": f1_score(y_true, y_pred, average="macro", zero_division=0),
                "tcs": tcs
            }

            video_metrics.append(metrics)

            logger.info(
                f"Video {video_name} | "
                f"Acc: {metrics['accuracy']:.3f} | "
                f"F1: {metrics['f1_score']:.3f} |"
                f"TCS: {metrics['tcs']:.3f} | "
                f"Prec: {metrics['precision']:.3f} | "
                f"Recall: {metrics['recall']:.3f}"
            )

        # =============================
        # MEDIA SULLO STREAM
        # =============================
        if not video_metrics:
            logger.error(f"Nessuna metrica calcolata per lo stream {stream_name}")
            return {}

        stream_metrics = {
            "stream": stream_name,
            "accuracy": np.mean([m["accuracy"] for m in video_metrics]),
            "precision": np.mean([m["precision"] for m in video_metrics]),
            "recall": np.mean([m["recall"] for m in video_metrics]),
            "f1_score": np.mean([m["f1_score"] for m in video_metrics]),
            "tcs": np.nanmean([m["tcs"] for m in video_metrics]),  
            "num_videos": len(video_metrics)
        }

        logger.info(
            f"STREAM {stream_name} | "
            f"Acc: {stream_metrics['accuracy']:.3f} | "
            f"F1: {stream_metrics['f1_score']:.3f} |"
            f"Prec: {stream_metrics['precision']:.3f} | "
            f"Recall: {stream_metrics['recall']:.3f} | "
        )

        return {
            "per_video": video_metrics,
            "stream_average": stream_metrics
        }
    
    import numpy as np

    def _emotion_dict_to_vector(self, emotions: dict, emotion_order: list) -> np.ndarray:
        """
        Converte dict emozioni -> vettore in ordine fisso.
        Normalizza in somma 1 (se non lo è già).
        """
        v = np.array([float(emotions.get(e, 0.0)) for e in emotion_order], dtype=np.float64)
        s = v.sum()
        if s <= 0:
            return v  # tutto zero (sarà gestito fuori)
        return v / s

    def _js_divergence(self, p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
        """
        Jensen-Shannon divergence (base e). Output in [0, ln(2)] se p,q sono distribuzioni.
        """
        p = np.clip(p, eps, 1.0); p = p / p.sum()
        q = np.clip(q, eps, 1.0); q = q / q.sum()
        m = 0.5 * (p + q)

        kl_pm = np.sum(p * np.log(p / m))
        kl_qm = np.sum(q * np.log(q / m))
        return 0.5 * (kl_pm + kl_qm)

    def _temporal_consistency_score(self, prob_seq: list[np.ndarray], weight_seq: list[float] | None = None) -> float:
        """
        TCS = 1 - media pesata delle JS divergence tra step consecutivi.
        Range tipico ~ [0,1]. Se seq corta -> NaN.
        """
        if len(prob_seq) < 2:
            return float("nan")

        js_vals = []
        w_vals = []

        for i in range(1, len(prob_seq)):
            p_prev = prob_seq[i - 1]
            p_curr = prob_seq[i]

            # salta se vettori non validi (tutto zero)
            if p_prev.sum() <= 0 or p_curr.sum() <= 0:
                continue

            js = self._js_divergence(p_curr, p_prev)
            js_vals.append(js)

            if weight_seq is not None:
                # peso tra i due istanti (media dei due)
                w_vals.append(0.5 * (weight_seq[i] + weight_seq[i - 1]))

        if not js_vals:
            return float("nan")

        if weight_seq is None:
            mean_js = float(np.mean(js_vals))
        else:
            w = np.array(w_vals, dtype=np.float64)
            if w.sum() <= 0:
                mean_js = float(np.mean(js_vals))
            else:
                mean_js = float(np.average(js_vals, weights=w))

        return 1.0 - mean_js

    def evaluate_fusion(
        self,
        stream_names: list[str],
        video_name_list: list[str]
    ) -> dict:
        """
        Valuta la fusione multimodale (late fusion - average).
        """

        logger = self.utils.setup_logger()
        logger.info(f"Evaluating fusion of streams: {stream_names}")

        result_json = pd.read_json(
            self.utils.config["Paths"]["complete_info_path"]
        )
        
        # Escludo il file json se presente
        video_name_list = [n for n in video_name_list if "json" not in n]

        video_metrics = []

        # =============================
        # LOOP PER VIDEO
        # =============================
        for video in video_name_list:
            video_name = video.split("/")[1]
            
            if video_name not in result_json:
                logger.warning(f"Video {video_name} non trovato.")
                continue

            y_true = []
            y_pred = []

            for ts in result_json[video_name]["time_slots"]:
                gt = ts["ground_truth"]

                modal_emotions = {
                    s: ts["modal"].get(s, {}).get("emotions", {})
                    for s in stream_names
                }

                fused_emotions = self.fuse_emotions_average(modal_emotions)

                if not fused_emotions:
                    continue

                pred = self._emotion_dict_to_label(fused_emotions)

                y_true.append(gt)
                y_pred.append(pred)

            if not y_true:
                logger.warning(f"Nessun time-slot valido per la fusione in {video_name}")
                continue

            metrics = {
                "video": video_name,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
                "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
                "f1_score": f1_score(y_true, y_pred, average="macro", zero_division=0),
            }

            video_metrics.append(metrics)

            logger.info(
                f"[Fusion] {video_name} | "
                f"Acc: {metrics['accuracy']:.3f} | "
                f"F1: {metrics['f1_score']:.3f} | "
                f"Prec: {metrics['precision']:.3f} | "
                f"Recall: {metrics['recall']:.3f}"
            )

        # =============================
        # MEDIA FINALE
        # =============================
        if not video_metrics:
            logger.error("Nessuna metrica calcolata per la fusione")
            return {}

        fusion_metrics = {
            "fusion_type": "late_average",
            "streams": stream_names,
            "accuracy": np.mean([m["accuracy"] for m in video_metrics]),
            "precision": np.mean([m["precision"] for m in video_metrics]),
            "recall": np.mean([m["recall"] for m in video_metrics]),
            "f1_score": np.mean([m["f1_score"] for m in video_metrics]),
            "num_videos": len(video_metrics)
        }

        logger.info(
            f"[Fusion FINAL] Acc: {fusion_metrics['accuracy']:.3f} | "
            f"F1: {fusion_metrics['f1_score']:.3f} | "
            f"Prec: {fusion_metrics['precision']:.3f} | "
            f"Recall: {fusion_metrics['recall']:.3f}"
        )

        return {
            "per_video": video_metrics,
            "fusion_average": fusion_metrics
        }

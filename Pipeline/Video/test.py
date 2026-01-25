import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Batch
from scipy.spatial import Delaunay

# ============================================================
# LANDMARK CHIAVE (MediaPipe FaceMesh)
# ============================================================
NOSE_TIP = 1
MOUTH_CENTER = 13
LEFT_EYE = 33
RIGHT_EYE = 263


# ============================================================
# PIPELINE: IMMAGINE → LANDMARK → FEATURE → GRAFO
# ============================================================
class FaceGraphPipeline:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def extract_face_landmarks(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

    def normalize_landmarks(self, coords):
        # Centramento sul naso
        nose_coords = coords[NOSE_TIP]
        centered_coords = coords - nose_coords

        # Scaling basato sulla distanza tra gli occhi
        dist_eyes = np.linalg.norm(coords[LEFT_EYE] - coords[RIGHT_EYE])
        normalized_coords = centered_coords / (dist_eyes + 1e-6)

        # Distanza dalla bocca
        mouth_coords = normalized_coords[MOUTH_CENTER]
        dist_to_mouth = np.linalg.norm(
            normalized_coords - mouth_coords, axis=1
        ).reshape(-1, 1)

        # Inclinazione locale rispetto al baricentro
        mean_pos = np.mean(normalized_coords, axis=0)
        local_inclination = normalized_coords - mean_pos

        # Feature finali: [x,y,z, dist_mouth, dx,dy,dz]
        features = np.hstack([
            normalized_coords,
            dist_to_mouth,
            local_inclination
        ])

        return torch.tensor(features, dtype=torch.float)

    def get_delaunay_edges(self, coords):
        tri = Delaunay(coords[:, :2])
        edges = set()

        for simplex in tri.simplices:
            edges.add(tuple(sorted((simplex[0], simplex[1]))))
            edges.add(tuple(sorted((simplex[1], simplex[2]))))
            edges.add(tuple(sorted((simplex[2], simplex[0]))))

        edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        return edge_index

    # ========================================================
    # VISUALIZZAZIONE LANDMARK NORMALIZZATI (DEBUG)
    # ========================================================
    def visualize_normalized_landmarks(self, image_path, normalized_coords, scale=150):
        image = cv2.imread(image_path)
        if image is None:
            return

        h, w, _ = image.shape
        cx, cy = w // 2, h // 2

        points = normalized_coords[:, :2].numpy()

        for (x, y) in points:
            px = int(cx + x * scale)
            py = int(cy + y * scale)
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(image, (px, py), 1, (0, 255, 0), -1)

        cv2.imshow("Normalized Face Landmarks (Debug)", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ============================================================
# MODELLO GNN: GATv2
# ============================================================
class EmotionGATv2(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()

        self.conv1 = GATv2Conv(num_node_features, 32, heads=4)
        self.conv2 = GATv2Conv(32 * 4, 64, heads=4)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(64 * 4, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x, edge_index, batch):
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.classifier(x)


# ============================================================
# MAIN
# ============================================================
def main():

    frame_paths =["CompletePipeline\\01-01-01-01-01-01-01.mp4\\Video\extractedFaceFrames\\face_ts1_fr3.jpg","CompletePipeline\\01-01-01-01-01-01-01.mp4\\Video\extractedFaceFrames\\face_ts1_fr6.jpg",
                   "CompletePipeline\\01-01-01-01-01-01-01.mp4\\Video\extractedFaceFrames\\face_ts1_fr9.jpg", "CompletePipeline\\01-01-01-01-01-01-01.mp4\\Video\extractedFaceFrames\\face_ts1_f12.jpg",
                   "CompletePipeline\\01-01-01-01-01-01-01.mp4\\Video\extractedFaceFrames\\face_ts3_fr75.jpg"]

    pipeline = FaceGraphPipeline()
    model = EmotionGATv2(num_node_features=7, num_classes=7)
    model.eval()

    graphs = []

    for path in frame_paths:
        landmarks = pipeline.extract_face_landmarks(path)
        if landmarks is None:
            continue

        node_features = pipeline.normalize_landmarks(landmarks)

        # 🔴 VISUAL DEBUG (commenta se non serve)
        pipeline.visualize_normalized_landmarks(
            path,
            node_features[:, :3],
            scale=150
        )

        edge_index = pipeline.get_delaunay_edges(landmarks)
        graphs.append(Data(x=node_features, edge_index=edge_index))

    if not graphs:
        print("Nessun volto rilevato.")
        return

    batch = Batch.from_data_list(graphs)

    with torch.no_grad():
        preds = model(batch.x, batch.edge_index, batch.batch)
        labels = torch.argmax(preds, dim=1)

    print("Emozioni predette per frame:", labels.tolist())


if __name__ == "__main__":
    main()

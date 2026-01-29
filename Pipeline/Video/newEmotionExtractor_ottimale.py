import os
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from scipy.spatial import Delaunay
from datasets import load_dataset
from torch_geometric.nn import GlobalAttention


# file con codice piu' complesso e aggiornato: con questa soluzione su 250 video ravdess
#  2026-01-28 20:45:31 INFO STREAM video | Acc: 0.860 | F1: 0.822 |Prec: 0.858 | Recall: 0.808 |

# OMG con 5 video di test
# 2026-01-28 21:23:56 INFO STREAM video | Acc: 0.405 | F1: 0.214 |Prec: 0.211 | Recall: 0.263 | 
# OMG con 20 video di test
# 2026-01-28 22:58:05 INFO STREAM video | Acc: 0.395 | F1: 0.188 |Prec: 0.204 | Recall: 0.258 | 

# ============================================================
# COSTANTI LANDMARK
# ============================================================
NOSE_TIP = 1
MOUTH_CENTER = 13
LEFT_EYE = 33
RIGHT_EYE = 263

# ============================================================
# PIPELINE DI PREPROCESSING LANDMARK
# ============================================================
class FaceGraphPipeline:
    def __init__(self):
        # Inizializza Mediapipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3
        )

    def normalize_landmarks(self, coords):
        """
        Normalizza i landmark:
        - centro sul naso
        - scala basata sulla distanza tra occhi
        - feature aggiuntive: distanza dal centro della bocca e distanza radiale dal centro
        """
        #  Centro sul naso
        nose = coords[NOSE_TIP]
        coords = coords - nose

        #  Scala sulla distanza tra occhi
        scale = np.linalg.norm(coords[LEFT_EYE] - coords[RIGHT_EYE])
        coords = coords / (scale + 1e-6)

        # Distanza dalla bocca 
        mouth = coords[MOUTH_CENTER]
        dist_to_mouth = np.linalg.norm(coords - mouth, axis=1, keepdims=True)

        # Distanza radiale dal centro del volto
        center = coords.mean(axis=0)
        radial_dist = np.linalg.norm(coords - center, axis=1, keepdims=True)



        #features = np.hstack([coords, dist_to_mouth, radial_dist])
        #return torch.tensor(features, dtype=torch.float)

        # AGGIUNTA PER MIGLIORARE ACCURACY 
        # --- ANGOLI LOCALI ---
        #neighbors = self.get_neighbors_from_delaunay(coords)
        #angles = self.compute_local_angles(coords, neighbors)

        # Feature finali
        features = np.hstack([
            coords,          # (x,y,z)
            dist_to_mouth,   # 1
            radial_dist     # 1
            #angles           # 1
        ])

        return torch.tensor(features, dtype=torch.float)

    def get_delaunay_edges(self, coords):
        """
        Costruisce i bordi tramite triangolazione di Delaunay
        """
        tri = Delaunay(coords[:, :2])
        edges = set()
        for simplex in tri.simplices:
            edges.add(tuple(sorted((simplex[0], simplex[1]))))
            edges.add(tuple(sorted((simplex[1], simplex[2]))))
            edges.add(tuple(sorted((simplex[2], simplex[0]))))
        edge_index = torch.tensor(list(edges), dtype=torch.long).t()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        return edge_index

    # AGGIUNTA PER AUMENTARE ACCURACY CON VICINI DELLA DELAUNAY
    def get_neighbors_from_delaunay(self, coords):
        tri = Delaunay(coords[:, :2])
        neighbors = [[] for _ in range(len(coords))]

        for simplex in tri.simplices:
            for i in range(3):
                for j in range(3):
                    if i != j:
                        a = simplex[i]
                        b = simplex[j]
                        if b not in neighbors[a]:
                            neighbors[a].append(b)

        return neighbors

    # AGGIUNTA PER AUMENTARE ACCURACY
    # Misura qunato è curvato localmente il volto in ogni landmark usando l'angolo tra due vicini nel grafo Delaunay
    def compute_local_angles(self, coords, neighbors):
        angles = []

        for i, nbrs in enumerate(neighbors):
            if len(nbrs) < 2:
                angles.append(0.0)
                continue

            p = coords[i]
            v1 = coords[nbrs[0]] - p
            v2 = coords[nbrs[1]] - p

            cos = np.dot(v1, v2) / (
                np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
            )
            angle = np.arccos(np.clip(cos, -1.0, 1.0))
            angles.append(angle)

        return np.array(angles).reshape(-1, 1)


# ============================================================
# MODELLO GATv2
# ============================================================
class EmotionGATv2(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_dim=64, heads=4):
        super().__init__()
        
        # --- Layer 1 ---
        self.conv1 = GATv2Conv(num_node_features, hidden_dim, heads=heads, dropout=0.2)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim * heads)

        # --- Layer 2 ---
        self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.2)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim * heads)

        # Aggiunto un layer 3  !!!!!!!!!!!!!!!!!
        # Concat=False qui per ridurre la dimensionalità prima del pooling
        self.conv3 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, dropout=0.2, concat=False)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim)

        # AGGIUNTA ATTENTION POOLING ED ATTENTION GLOBALE !!!!!!!!
        # --- ATTENTION POOLING ---
        # Questa piccola rete decide quanto è importante ogni nodo
        gate_nn = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(32, 1)
        )

        # Attention Globale per permettere alla rete di imparare quali nodi guardare 
        self.pool = GlobalAttention(gate_nn=gate_nn)

        # --- Classifier ---
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 128),
            torch.nn.Mish(),  # Attivazione moderna Mish !!!!!!!
            torch.nn.Dropout(0.4),
            torch.nn.Linear(128, 64),
            torch.nn.Mish(),
            torch.nn.Dropout(0.3), # aggiunto un nuovo strato di dropout 
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, x, edge_index, batch):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.mish(self.bn1(x))
        x = F.dropout(x, p=0.2, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.mish(self.bn2(x))
        x = F.dropout(x, p=0.2, training=self.training)

        # Layer 3
        x = self.conv3(x, edge_index)
        x = self.bn3(x) # Niente attivazione qui, lasciamo che il pooling gestisca

        # Pooling Attentivo (restituisce [batch_size, hidden_dim])
        x_pooled = self.pool(x, batch)

        return self.classifier(x_pooled)


class NewEmotionExtractor:
    def __init__(self, model_path):
        self.pipeline = FaceGraphPipeline()

        self.model = EmotionGATv2(
            num_node_features=5,
            num_classes=7
        )

        ckpt = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        print(f"[INFERENCE] Modello caricato da {model_path}")

    def extractFaceEmotion(self, image, return_all=True):
        results = self.pipeline.mp_face_mesh.process(image)

        if not results.multi_face_landmarks:
            return None

        landmarks = np.array([
            [lm.x, lm.y, lm.z]
            for lm in results.multi_face_landmarks[0].landmark
        ])

        x = self.pipeline.normalize_landmarks(landmarks)
        edge_index = self.pipeline.get_delaunay_edges(landmarks)

        data = Data(x=x, edge_index=edge_index)
        batch = torch.zeros(data.x.size(0), dtype=torch.long)

        with torch.no_grad():
            logits = self.model(data.x, data.edge_index, batch)
            probs = torch.softmax(logits, dim=1)

        # libera riferimenti
        del data, batch, logits
        #print(probs)
        #print(probs.size())
        return probs.squeeze(0)

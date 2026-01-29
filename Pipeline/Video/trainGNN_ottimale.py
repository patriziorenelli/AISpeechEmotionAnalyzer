import os
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GlobalAttention
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from scipy.spatial import Delaunay
from datasets import load_dataset
from sklearn.utils.class_weight import compute_class_weight

# QUESTO FILE CONTIENE IL CODICE AGGIORNATO PIU' COMPLESSO 
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

# ============================================================
# DATASET RAF-DB
# ============================================================
class RAFDBGraphDataset(Dataset):
    def __init__(self, pipeline, split="train"):
        self.pipeline = pipeline
        # Carica solo lo split specificato (train/test)
        self.dataset = load_dataset("deanngkl/raf-db-7emotions", split=split)
        self.split_name = split
        print(f"[DATASET] RAF-DB {self.split_name} | samples={len(self.dataset)}")
        print(".....split loaded....")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"].convert("RGB")
        #image = np.array(sample["image"])
        image = np.array(image)


        results = self.pipeline.mp_face_mesh.process(image)
        if not results.multi_face_landmarks:
            #print(f"[DATASET] NESSUN VOLTO in idx={idx}, salto al successivo")
            return self.__getitem__((idx + 1) % len(self))

        #print(f"[DATASET] VOLTO in idx={idx} PROCESSATO")

        landmarks = np.array([[lm.x, lm.y, lm.z]
                              for lm in results.multi_face_landmarks[0].landmark])

        x = self.pipeline.normalize_landmarks(landmarks)
        edge_index = self.pipeline.get_delaunay_edges(landmarks)

        if idx % 50 == 0:
            print(f"[DATASET] Processati {idx+1}/{len(self.dataset)} campioni")

        return Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor(sample["label"], dtype=torch.long)
        )

# ============================================================
# TRAINER CON CHECKPOINT
# ============================================================
class EmotionTrainer:
    def __init__(self, model, lr=1e-3, device="cpu", class_weights=None):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=5e-4
        )
        #if class_weights is not None: # Usiamo i pesi delle varie classi all'interno della loss 
        #    self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
        #else:
        #    self.criterion = torch.nn.CrossEntropyLoss() # O LabelSmoothingCrossEntropy

    def save_checkpoint(self, path, epoch, best_acc):
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_acc": best_acc
        }, path)
        print(f"[CHECKPOINT] Salvato: {path}")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        print(f"[CHECKPOINT] Ripreso da epoch {ckpt['epoch']} | BestAcc={ckpt['best_acc']:.3f}")
        return ckpt["epoch"], ckpt["best_acc"]

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for i, batch in enumerate(loader, 1):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            # loss = self.criterion(out, batch.y) # Usando la loss con pesi di classe
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            # Accuracy batch
            pred = out.argmax(dim=1)
            correct = (pred == batch.y).sum().item()
            total_correct += correct
            total_samples += batch.y.size(0)
            batch_acc = correct / batch.y.size(0)

            if i % 10 == 0 or i == len(loader):
                print(f"[TRAIN] Batch {i}/{len(loader)} | Loss={loss.item():.4f} | Acc batch={batch_acc:.3f}")

        epoch_acc = total_correct / total_samples
        print(f"[TRAIN] Accuracy totale epoch: {epoch_acc:.3f}")
        return total_loss / len(loader)


    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        correct, total = 0, 0
        for i, batch in enumerate(loader, 1):
            batch = batch.to(self.device)
            out = self.model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)

            if i % 5 == 0 or i == len(loader):
                print(f"[VAL] Batch {i}/{len(loader)} | Acc parziale={correct/total:.3f}")

        return correct / total

    def fit(self, train_loader, val_loader, epochs=20, ckpt_dir="checkpoints"):
        os.makedirs(ckpt_dir, exist_ok=True)
        best_acc = 0.0

        for epoch in range(1, epochs + 1):
            print(f"\n--- Epoch {epoch}/{epochs} ---")
            loss = self.train_epoch(train_loader)
            acc = self.evaluate(val_loader)

            print(f"[EPOCH {epoch}/{epochs}] Loss={loss:.4f} | ValAcc={acc:.3f}")

            self.save_checkpoint(
                os.path.join(ckpt_dir, f"epoch_{epoch}.pt"),
                epoch,
                best_acc
            )

            if acc > best_acc:
                best_acc = acc
                self.save_checkpoint(
                    os.path.join(ckpt_dir, "best_model.pt"),
                    epoch,
                    best_acc
                )
                print(f"🔥 NEW BEST MODEL | ValAcc={best_acc:.3f}")

# ============================================================
# TRAINING
# ============================================================
def train_model(class_weights=None):
    pipeline = FaceGraphPipeline()
    dataset = RAFDBGraphDataset(pipeline, split="train")

    # Split train/val 90/10
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    print(".......PRIMA DEL MODELLO........")
    model = EmotionGATv2(num_node_features=5, num_classes=7)
    # model = EmotionGATv2(num_node_features=6, num_classes=7)

    trainer = EmotionTrainer(
        model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        class_weights=class_weights
    )
    print(".......DOPO IL MODELLO........")
    trainer.fit(train_loader, val_loader, epochs=20)

# ============================================================
# INFERENZA SU FRAME SINGOLI
# ============================================================
def inference_on_frames(frame_paths, model_path="checkpoints/best_model.pt"):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Il modello non esiste: '{model_path}'. Assicurati di aver eseguito il training e generato il checkpoint.")

    pipeline = FaceGraphPipeline()
    model = EmotionGATv2(num_node_features=5, num_classes=7)
    #model = EmotionGATv2(num_node_features=6, num_classes=7)
    ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[INFERENCE] Modello caricato da {model_path}")

    graphs = []
    for idx, path in enumerate(frame_paths):
        image = cv2.imread(path)
        if image is None:
            print(f"[INFERENCE] Immagine non trovata: {path}")
            continue

        results = pipeline.mp_face_mesh.process(image)
        if not results.multi_face_landmarks:
            print(f"[INFERENCE] Nessun volto rilevato in: {path}")
            continue

        landmarks = np.array([[lm.x, lm.y, lm.z]
                              for lm in results.multi_face_landmarks[0].landmark])
        x = pipeline.normalize_landmarks(landmarks)
        edge_index = pipeline.get_delaunay_edges(landmarks)
        graphs.append(Data(x=x, edge_index=edge_index))
        print(f"[INFERENCE] Volto rilevato in {path}")

    if not graphs:
        print("Nessun volto rilevato in nessun frame.")
        return

    batch = Batch.from_data_list(graphs)
    with torch.no_grad():
        preds = model(batch.x, batch.edge_index, batch.batch).argmax(dim=1)

    print("[INFERENCE] Emozioni predette:", preds.tolist())


# Funzione per calcolare i pesi di classe per cercare di ridurre lo sbilanciamento tra classi nel dataset 
def class_weights_extraction():
    hf_dataset = load_dataset("deanngkl/raf-db-7emotions", split="train")

    # Estrai le labels come array
    all_labels = np.array(hf_dataset["label"])

    # Calcola pesi bilanciati
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(all_labels),
        y=all_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print("Class weights:", class_weights)
    return class_weights

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":


    class_weights = class_weights_extraction()

    train_model(class_weights)

    frame_paths = [
        "CompletePipeline\\01-01-01-01-01-01-01.mp4\\Video\\extractedFaceFrames\\face_ts1_fr3.jpg",
        "CompletePipeline\\01-01-01-01-01-01-01.mp4\\Video\\extractedFaceFrames\\face_ts1_fr6.jpg"
    ]
    #inference_on_frames(frame_paths)

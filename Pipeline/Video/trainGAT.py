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
        nose = coords[NOSE_TIP]
        coords = coords - nose

        scale = np.linalg.norm(coords[LEFT_EYE] - coords[RIGHT_EYE])
        coords = coords / (scale + 1e-6)

        mouth = coords[MOUTH_CENTER]
        dist_to_mouth = np.linalg.norm(coords - mouth, axis=1, keepdims=True)

        center = coords.mean(axis=0)
        radial_dist = np.linalg.norm(coords - center, axis=1, keepdims=True)

        features = np.hstack([coords, dist_to_mouth, radial_dist])
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

# ============================================================
# MODELLO GATv2
# ============================================================
class EmotionGATv2(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        # --- Layer GATv2 con normalizzazione batch ---
        self.conv1 = GATv2Conv(num_node_features, 32, heads=4, dropout=0.2)
        self.bn1 = torch.nn.BatchNorm1d(32 * 4)

        self.conv2 = GATv2Conv(32 * 4, 64, heads=4, dropout=0.2)
        self.bn2 = torch.nn.BatchNorm1d(64 * 4)

        # --- Shortcut residuo ---
        self.res1 = torch.nn.Linear(32 * 4, 64 * 4)

        # --- Classifier finale ---
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(128 * 4, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x, edge_index, batch):
        # --- Blocco 1 ---
        x = self.conv1(x, edge_index)
        x1 = F.elu(self.bn1(x))  # output: 128 feature
        x1 = F.dropout(x1, p=0.2, training=self.training)

        # --- Blocco 2 con residuo ---
        out = self.conv2(x1, edge_index)
        out = self.bn2(out)
        res = self.res1(x1)
        x2 = F.elu(out + res)  # output: 256 feature

        # --- Global Pooling ---
        mean_pool = global_mean_pool(x2, batch)
        max_pool = global_max_pool(x2, batch)
        x_pooled = torch.cat([mean_pool, max_pool], dim=1)  # 512 feature

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
    def __init__(self, model, lr=1e-3, device="cpu"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=5e-4
        )

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
def train_model():
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
    trainer = EmotionTrainer(
        model,
        device="cuda" if torch.cuda.is_available() else "cpu"
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

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    train_model()

    frame_paths = [
        "CompletePipeline\\01-01-01-01-01-01-01.mp4\\Video\\extractedFaceFrames\\face_ts1_fr3.jpg",
        "CompletePipeline\\01-01-01-01-01-01-01.mp4\\Video\\extractedFaceFrames\\face_ts1_fr6.jpg"
    ]
    #inference_on_frames(frame_paths)

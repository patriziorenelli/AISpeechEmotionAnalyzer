import os
import cv2
import numpy as np
from typing import List
import mediapipe as mp
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.neighbors import NearestNeighbors

# VEDERE SE LE FEAUTURE ESTRATTE HANNO SENSO E COME UTILIZZARLE 
# Provare ad eseguire GAT sul video sample di RAVDESS e usare gli embedding per qualche CNN pre trainata apposta o classificatore SVM

def graph_collate_fn(batch):
    """
    Evita che PyTorch provi a collare oggetti torch_geometric.data.Data.
    Restituisce la lista così com'è:
    batch = [ sequence_di_frame ]
    """
    return batch

# ----------------------
# 1) COSTRUZIONE GRAFO KNN
# ----------------------
def build_knn_edges(pts, k=6):
    """
    pts: numpy array or torch tensor of shape [N, 2] (x,y normalized)
    returns: torch.LongTensor edge_index shape [2, E]
    """
    if isinstance(pts, torch.Tensor):
        pts_np = pts.cpu().numpy()
    else:
        pts_np = np.asarray(pts)

    if pts_np.ndim == 1:
        pts_np = pts_np.reshape(-1, 2)

    neigh = NearestNeighbors(n_neighbors=min(k + 1, len(pts_np))).fit(pts_np)
    _, idxs = neigh.kneighbors(pts_np)

    edges = []
    for i in range(len(pts_np)):
        for j in idxs[i][1:]:
            edges.append([i, int(j)])

    if len(edges) == 0:
        # fallback: no edges (single node)
        return torch.zeros((2, 0), dtype=torch.long)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

# ----------------------
# 2) VISUALIZZAZIONE DEBUG (opzionale)
# ----------------------
def visualize_graph(pts, edge_index):
    import matplotlib.pyplot as plt
    pts = np.asarray(pts)
    plt.figure(figsize=(6, 6))
    plt.scatter(pts[:, 0], pts[:, 1], s=6, c='red')
    for i, j in edge_index.t().cpu().numpy().T:
        x = [pts[int(i), 0], pts[int(j), 0]]
        y = [pts[int(i), 1], pts[int(j), 1]]
        plt.plot(x, y, color='black', linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.title("Grafo Landmark (KNN)")
    plt.show()

# ----------------------
# 3) DISEGNO SUL FRAME
# ----------------------
def draw_graph_on_frame(frame, pts, edge_index):
    """
    Disegna landmark + archi sul frame OpenCV.
    pts: array Nx2 normalizzate
    edge_index: torch tensor [2, E]
    """
    frame_out = frame.copy()
    h, w = frame.shape[:2]
    if edge_index.numel() > 0:
        for i, j in edge_index.t().cpu().numpy():
            x1, y1 = int(pts[int(i), 0] * w), int(pts[int(i), 1] * h)
            x2, y2 = int(pts[int(j), 0] * w), int(pts[int(j), 1] * h)
            cv2.line(frame_out, (x1, y1), (x2, y2), (0, 255, 0), 1)
    for p in pts:
        x, y = int(p[0] * w), int(p[1] * h)
        cv2.circle(frame_out, (x, y), 2, (0, 0, 255), -1)
    return frame_out

# ----------------------
# 4) ESTRAZIONE LANDMARK (MediaPipe)
# ----------------------
mp_face_mesh = mp.solutions.face_mesh

def extract_landmarks_from_frame(frame):
    """
    Restituisce Nx2 array (x,y) normalizzati oppure None
    """
    # Uso il context manager per ogni frame per semplicità e robustezza
    with mp_face_mesh.FaceMesh(static_image_mode=False,
                               max_num_faces=1,
                               refine_landmarks=True,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as face_mesh:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if res.multi_face_landmarks:
            pts = []
            for lm in res.multi_face_landmarks[0].landmark:
                pts.append([lm.x, lm.y])
            return np.array(pts, dtype=np.float32)
    return None

# ----------------------
# 5) DATASET
# ----------------------
class FaceGraphSequenceDataset(Dataset):
    def __init__(self, video_paths: List[str], seq_len: int = 16, k: int = 6, save_graph_video: bool = False):
        self.video_paths = video_paths
        self.seq_len = seq_len
        self.k = k
        self.save_graph_video = save_graph_video

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Impossibile aprire video: {path}")

        frames_pts = []

        # setup writer se richiesto
        writer = None
        if self.save_graph_video:
            out_path = os.path.splitext(path)[0] + "_graph.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        ret, frame = cap.read()
        while ret:
            pts = extract_landmarks_from_frame(frame)
            if pts is not None and len(pts) > 0:
                # edge_index come torch tensor
                edge_index = build_knn_edges(pts[:, :2], k=self.k)

                if writer is not None:
                    frame_out = draw_graph_on_frame(frame, pts, edge_index)
                    writer.write(frame_out)

                frames_pts.append(np.array(pts, dtype=np.float32))
            # leggi prossimo
            ret, frame = cap.read()

        if writer is not None:
            writer.release()
        cap.release()

        if len(frames_pts) == 0:
            raise ValueError(f"Nessun volto rilevato in {path}")

        # pad/trim alla seq_len
        if len(frames_pts) < self.seq_len:
            frames_pts = frames_pts + [frames_pts[-1].copy()] * (self.seq_len - len(frames_pts))
        frames_pts = frames_pts[:self.seq_len]

        # costruisci lista di Data (PyG)
        seq_graphs = []
        for pts in frames_pts:
            pts_t = torch.tensor(pts, dtype=torch.float32)  # [N,2]
            edge_index = build_knn_edges(pts_t[:, :2], k=self.k)  # torch [2,E]
            x = pts_t  # node features: (x,y) ; puoi estendere con z/delta
            graph = Data(x=x, edge_index=edge_index)
            seq_graphs.append(graph)

        return seq_graphs

# ----------------------
# 6) MODELLO: GAT per frame + LSTM per sequenza
# ----------------------
class GATEncoder(nn.Module):
    def __init__(self, in_channels=2, hidden=32, out_channels=128, heads=4, dropout=0.1):
        super().__init__()
        self.layer1 = GATConv(in_channels, hidden, heads=heads, dropout=dropout)
        self.layer2 = GATConv(hidden * heads, out_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        # x: [N, in_channels], edge_index: [2, E]
        h = self.layer1(x, edge_index).relu()
        h = self.layer2(h, edge_index)
        # global mean pooling: batch of size 1, quindi costruisco batch zeros
        batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        pooled = global_mean_pool(h, batch)  # [1, out_channels]
        return pooled.squeeze(0)  # [out_channels]

class ExpressionClassifier(nn.Module):
    def __init__(self, gat_out=128, lstm_hidden=128, num_classes=7):
        super().__init__()
        self.encoder = GATEncoder(in_channels=2, out_channels=gat_out)
        self.lstm = nn.LSTM(gat_out, lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, graphs_sequence: List[Data]):
        """
        graphs_sequence: list length seq_len of Data objects
        returns: logits [1, num_classes], embeddings_list [seq_len tensors of shape (gat_out,)]
        """
        device = next(self.parameters()).device
        encodings = []
        for g in graphs_sequence:
            x = g.x.to(device)
            edge_index = g.edge_index.to(device)
            z = self.encoder(x, edge_index)  # [gat_out]
            encodings.append(z)
        seq = torch.stack(encodings, dim=0).unsqueeze(0)  # [1, seq_len, gat_out]
        out, _ = self.lstm(seq)  # out [1, seq_len, lstm_hidden]
        logits = self.fc(out[:, -1, :])  # [1, num_classes]
        return logits, encodings

# ----------------------
# 7) MAIN
# ----------------------
def main():
    video_files = ["faces_video.mp4"]  # metti i tuoi file qui
    seq_len = 16
    k = 6
    save_graph_video = True  # salva *_graph.mp4 accanto agli originali

    dataset = FaceGraphSequenceDataset(video_files, seq_len=seq_len, k=k, save_graph_video=save_graph_video)
    dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=graph_collate_fn
)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExpressionClassifier(gat_out=128, lstm_hidden=128, num_classes=7).to(device)
    model.eval()  # modalità inference per estrazione embedding

    # Itera i video (DataLoader restituisce batch di 1 elemento -> lista di grafi)
    for idx, batch in enumerate(dataloader):
        seq_graphs = batch[0]
        logits, embeddings = model(seq_graphs)

        # embeddings: lista di tensor [gat_out]
        emb_arr = np.stack([e.cpu().detach().numpy() for e in embeddings], axis=0)  # [seq_len, gat_out]

        # Stampa embedding di ogni frame
        for i, e in enumerate(embeddings):
            e_np = e.detach().cpu().numpy()
            print(f"  Frame {i}: embedding shape {e_np.shape}")
            print(e_np)

        # nome file per salvataggio
        video_path = video_files[idx]
        base = os.path.splitext(os.path.basename(video_path))[0]
        emb_filename = f"{base}_embeddings.npy"
        np.save(emb_filename, emb_arr)

        print(f"Video: {video_path}")
        print(f" - logits: {logits.detach().cpu().numpy()}")
        print(f" - saved embeddings shape: {emb_arr.shape} -> {emb_filename}")


    print("Elaborazione completata.")

if __name__ == "__main__":
    main()

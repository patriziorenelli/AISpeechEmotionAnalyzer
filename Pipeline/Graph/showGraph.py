import numpy as np
import matplotlib.pyplot as plt

# Dati originali
data = {
    'angry': 0.290258186,
    'disgust': 4.174178e-12,
    'fear': 0.1868011517,
    'happy': 5.447066e-08,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
}

labels = list(data.keys())
values = np.array(list(data.values()))

# Chiudo il poligono
values = np.append(values, values[0])

# Angoli per ogni punto del poligono
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
angles = np.append(angles, angles[0])

# Creazione figura
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

# Disegna l'area SENZA vincolare i punti agli assi
ax.plot(angles, values, color="#4AA89E", linewidth=2)
ax.fill(angles, values, color="#7FD4C1", alpha=0.6)

# Etichette degli assi
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=12)

# Limite radiale massimo automaticamente corretto
ax.set_ylim(0, max(values)*1.1)

# Griglia
ax.grid(color="gray", alpha=0.3)

plt.title("Emotion Radar Chart")
plt.tight_layout()
plt.show()

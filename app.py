import deeplake

# Carica il dataset remoto
ds = deeplake.load("hub://activeloop/ravdess-emotional-speech-audio")
ds.visualize()
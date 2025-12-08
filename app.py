
'''
import deeplake


# Carica il dataset remoto
ds = deeplake.load("hub://activeloop/ravdess-emotional-speech-audio")
ds.visualize()
'''

from datasets import load_dataset
ds = load_dataset("orvile/ravdess-dataset", streaming=True)

ds.visualize()
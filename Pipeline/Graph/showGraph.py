import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2

# Dati originali
data = {
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
}

# Colori fissi per emozione
color_map = {
    'neutral': 'gray',
    'disgust': 'green',
    'fear': 'purple',
    'happy': 'orange',
    'sad': 'black',
    'surprise': 'blue',
    'angry': 'red'
}

# lista delle emozioni con le medie per ogni singolo time_slot
data_list = [ {
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},  {
    'angry': 0.390258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},{
    'angry': 0.090258186,
    'disgust': 0.8174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},  {
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
}, {
    'angry': 0.290258186,
    'disgust': 0.8174178,
    'fear': 0.1868011517,
    'happy': 0.047066,
    'sad': 0.81354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
}, {
    'angry': 0.020258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.51354493,
    'surprise': 0.0049645943,
    'neutral': 0.2544311
}, {
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},{
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},  {
    'angry': 0.390258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},{
    'angry': 0.090258186,
    'disgust': 0.8174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},  {
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
}, {
    'angry': 0.290258186,
    'disgust': 0.8174178,
    'fear': 0.1868011517,
    'happy': 0.047066,
    'sad': 0.81354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
}, {
    'angry': 0.020258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.51354493,
    'surprise': 0.0049645943,
    'neutral': 0.2544311
}, {
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},{
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},  {
    'angry': 0.390258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},{
    'angry': 0.090258186,
    'disgust': 0.8174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},  {
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
}, {
    'angry': 0.290258186,
    'disgust': 0.8174178,
    'fear': 0.1868011517,
    'happy': 0.047066,
    'sad': 0.81354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
}, {
    'angry': 0.020258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.51354493,
    'surprise': 0.0049645943,
    'neutral': 0.2544311
}, {
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},{
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},  {
    'angry': 0.390258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},{
    'angry': 0.090258186,
    'disgust': 0.8174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},  {
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
}, {
    'angry': 0.290258186,
    'disgust': 0.8174178,
    'fear': 0.1868011517,
    'happy': 0.047066,
    'sad': 0.81354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
}, {
    'angry': 0.020258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.51354493,
    'surprise': 0.0049645943,
    'neutral': 0.2544311
}, {
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},{
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},  {
    'angry': 0.390258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},{
    'angry': 0.090258186,
    'disgust': 0.8174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},  {
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
}, {
    'angry': 0.290258186,
    'disgust': 0.8174178,
    'fear': 0.1868011517,
    'happy': 0.047066,
    'sad': 0.81354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
}, {
    'angry': 0.020258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.51354493,
    'surprise': 0.0049645943,
    'neutral': 0.2544311
}, {
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},{
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},  {
    'angry': 0.390258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},{
    'angry': 0.090258186,
    'disgust': 0.8174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},  {
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
}, {
    'angry': 0.290258186,
    'disgust': 0.8174178,
    'fear': 0.1868011517,
    'happy': 0.047066,
    'sad': 0.81354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
}, {
    'angry': 0.020258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.51354493,
    'surprise': 0.0049645943,
    'neutral': 0.2544311
}, {
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},{
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},  {
    'angry': 0.390258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},{
    'angry': 0.090258186,
    'disgust': 0.8174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},  {
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
}, {
    'angry': 0.290258186,
    'disgust': 0.8174178,
    'fear': 0.1868011517,
    'happy': 0.047066,
    'sad': 0.81354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
}, {
    'angry': 0.020258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.51354493,
    'surprise': 0.0049645943,
    'neutral': 0.2544311
}, {
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},{
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},  {
    'angry': 0.390258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},{
    'angry': 0.090258186,
    'disgust': 0.8174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},  {
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
}, {
    'angry': 0.290258186,
    'disgust': 0.8174178,
    'fear': 0.1868011517,
    'happy': 0.047066,
    'sad': 0.81354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
}, {
    'angry': 0.020258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.51354493,
    'surprise': 0.0049645943,
    'neutral': 0.2544311
}, {
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},{
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},  {
    'angry': 0.390258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},{
    'angry': 0.090258186,
    'disgust': 0.8174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},  {
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
}, {
    'angry': 0.290258186,
    'disgust': 0.8174178,
    'fear': 0.1868011517,
    'happy': 0.047066,
    'sad': 0.81354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
}, {
    'angry': 0.020258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.51354493,
    'surprise': 0.0049645943,
    'neutral': 0.2544311
}, {
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},{
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},  {
    'angry': 0.390258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},{
    'angry': 0.090258186,
    'disgust': 0.8174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},  {
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
}, {
    'angry': 0.290258186,
    'disgust': 0.8174178,
    'fear': 0.1868011517,
    'happy': 0.047066,
    'sad': 0.81354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
}, {
    'angry': 0.020258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.51354493,
    'surprise': 0.0049645943,
    'neutral': 0.2544311
}, {
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},{
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},  {
    'angry': 0.390258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},{
    'angry': 0.090258186,
    'disgust': 0.8174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},  {
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
}, {
    'angry': 0.290258186,
    'disgust': 0.8174178,
    'fear': 0.1868011517,
    'happy': 0.047066,
    'sad': 0.81354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
}, {
    'angry': 0.020258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.51354493,
    'surprise': 0.0049645943,
    'neutral': 0.2544311
}, {
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},{
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},  {
    'angry': 0.390258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},{
    'angry': 0.090258186,
    'disgust': 0.8174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},  {
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
}, {
    'angry': 0.290258186,
    'disgust': 0.8174178,
    'fear': 0.1868011517,
    'happy': 0.047066,
    'sad': 0.81354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
}, {
    'angry': 0.020258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.51354493,
    'surprise': 0.0049645943,
    'neutral': 0.2544311
}, {
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},{
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},  {
    'angry': 0.390258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},{
    'angry': 0.090258186,
    'disgust': 0.8174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},  {
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
}, {
    'angry': 0.290258186,
    'disgust': 0.8174178,
    'fear': 0.1868011517,
    'happy': 0.047066,
    'sad': 0.81354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
}, {
    'angry': 0.020258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.51354493,
    'surprise': 0.0049645943,
    'neutral': 0.2544311
}, {
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},{
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},  {
    'angry': 0.390258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},{
    'angry': 0.090258186,
    'disgust': 0.8174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},  {
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
}, {
    'angry': 0.290258186,
    'disgust': 0.8174178,
    'fear': 0.1868011517,
    'happy': 0.047066,
    'sad': 0.81354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
}, {
    'angry': 0.020258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.51354493,
    'surprise': 0.0049645943,
    'neutral': 0.2544311
}, {
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},{
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},  {
    'angry': 0.390258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},{
    'angry': 0.090258186,
    'disgust': 0.8174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},  {
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
}, {
    'angry': 0.290258186,
    'disgust': 0.8174178,
    'fear': 0.1868011517,
    'happy': 0.047066,
    'sad': 0.81354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
}, {
    'angry': 0.020258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.51354493,
    'surprise': 0.0049645943,
    'neutral': 0.2544311
}, {
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},{
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},  {
    'angry': 0.390258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},{
    'angry': 0.090258186,
    'disgust': 0.8174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
},  {
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
}, {
    'angry': 0.290258186,
    'disgust': 0.8174178,
    'fear': 0.1868011517,
    'happy': 0.047066,
    'sad': 0.81354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
}, {
    'angry': 0.020258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.51354493,
    'surprise': 0.0049645943,
    'neutral': 0.2544311
}, {
    'angry': 0.290258186,
    'disgust': 0.4174178,
    'fear': 0.1868011517,
    'happy': 0.5447066,
    'sad': 0.01354493,
    'surprise': 0.0049645943,
    'neutral': 0.5044311
}]


def graph_r_theta_plot():
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


def histogram_plot():
    sorted_items = sorted(data.items(), key=lambda x: x[1])
    emotions, values = zip(*sorted_items)
    colors = [color_map[e] for e in emotions]

    plt.figure(figsize=(8, 5))
    plt.barh(emotions, values, color=colors)

    plt.xlabel("Probabilità")
    plt.title("Distribuzione delle emozioni (ordinate)")
    plt.tight_layout()
    plt.show()

# Per mostrare l'evoluzione delle emozioni a livello temporale
def line_plot():
    emotions = data_list[0].keys()

    plt.figure(figsize=(10, 5))

    for e in emotions:
        plt.plot([frame[e] for frame in data_list], label=e)

    plt.xlabel("Frame / Tempo")
    plt.ylabel("Probabilità")
    plt.title("Andamento temporale delle emozioni")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Per mostrare come si evolvono le emozioni nel tempo 
def temporal_graph():
    emotions = list(data_list[0].keys())
    colors = [color_map[e] for e in emotions]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(emotions, [0]*len(emotions), color=colors)

    ax.set_xlim(0, 1)
    ax.set_xlabel("Probabilità")
    ax.set_title("Evoluzione delle emozioni")

    def update(frame_idx):
        frame = data_list[frame_idx]

        for bar, emotion in zip(bars, emotions):
            bar.set_width(frame[emotion])

        ax.set_title(f"Evoluzione delle emozioni – frame {frame_idx}")

        if frame_idx == len(data_list) - 1:
            plt.close(fig)

        return bars

    ani = FuncAnimation(
        fig,
        update,
        frames=len(data_list),
        interval=300,
        repeat=False      
    )

    plt.tight_layout()
    plt.show()

# Funzione che mostra il video e la media delle emozioni in modo sincrono 
#   -> per ora è soltanto la media (da capire se fare pesata) dei score ottenuti dalle 3 pipeline separate
#   -> si può pensare di prendere come input il nome file e il file json,
#   -> recuperare dal file le info sui singoli time slot che rappresentano un secondo, calcolare le medie, scaricare il video mostrare il grafico sotto
#   NON E' NECESSARIO SCARICARE IL VIDEO SE SI FA PER I SINGOLI VIDEO PRIMA DI ELIMINARE I FILE  

def temporal_graph_with_video(video_path, data_list, color_map):

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Errore apertura video"

    # ===== Emozioni =====
    emotions = list(data_list[0].keys())
    colors = [color_map[e] for e in emotions]

    fig, (ax_video, ax_bar) = plt.subplots(
        1, 2, figsize=(12, 5),
        gridspec_kw={'width_ratios': [1.2, 1]}
    )

    # --- Video subplot ---
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = ax_video.imshow(frame)
    ax_video.axis("off")
    ax_video.set_title("Video in analisi")

    # --- Istogramma subplot ---
    bars = ax_bar.barh(emotions, [0]*len(emotions), color=colors)
    ax_bar.set_xlim(0, 1)
    ax_bar.set_xlabel("Probabilità")
    ax_bar.set_title("Emozioni")

    # ===== Update =====
    def update(frame_idx):

        # --- Aggiorna video ---
        ret, frame = cap.read()
        if not ret:
            plt.close(fig)
            return bars

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im.set_data(frame)

        # --- Aggiorna istogramma ---
        emotion_frame = data_list[frame_idx]
        for bar, emotion in zip(bars, emotions):
            bar.set_width(emotion_frame[emotion])

        ax_bar.set_title(f"Emozioni – frame {frame_idx}")

        # STOP automatico
        if frame_idx == len(data_list) - 1:
            cap.release()
            plt.close(fig)

        return bars

    ani = FuncAnimation(
        fig,
        update,
        frames=min(len(data_list), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))),
        interval=30,       # ~33ms ≈ 30 FPS
        repeat=False
    )

    plt.tight_layout()
    plt.show()


temporal_graph_with_video("C:/Users/aless/OneDrive/Desktop/AiSpeech/CampioniVideo/faces_video_graph.mp4", data_list, color_map )


#temporal_graph()
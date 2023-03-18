from gensim.models import KeyedVectors
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
import requests
from PIL import Image
import random
import string
import re
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')


df = pd.read_csv('description.csv')
df = df[0:1000]

print(df[0])

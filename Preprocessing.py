import nltk
from textblob import TextBlob
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('punkt_tab') 
data = open("file.txt", encoding="utf-8").read()
tokenizer = RegexpTokenizer(r'\w+')
words = tokenizer.tokenize(data)
print(words[:30])
spell_corrected = [str(TextBlob(i).correct()) for i in words]
print(spell_corrected[:10])
new_text = " ".join(spell_corrected)
print(new_text)
tags = nltk.pos_tag(spell_corrected)
print(tags)
stop_words = set(stopwords.words("english"))
cleaned = [w for w in spell_corrected if w.lower() not in stop_words]
print(cleaned[:20])
stem = SnowballStemmer("english")
lemma = WordNetLemmatizer()
stem_out = [stem.stem(w) for w in cleaned]
lemma_out = [lemma.lemmatize(w) for w in cleaned]
print(stem_out[:20])
print(lemma_out[:20])
sent_list = sent_tokenize(new_text)
print(len(sent_list))

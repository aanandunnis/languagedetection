import joblib as jbl
import nltk
import pandas as pd
import string


model = jbl.load('finalized_model.sav')

inp = input("Enter a sentence: ")
inp = inp.lower()
x = inp.split()
out = []

for i in x:
  table = str.maketrans('', '', string.punctuation)
  out.append(model.classify_many([{'word':i.translate(table)}]))




df = pd.DataFrame(out)



l = {('bg',): 'Bulgarian',
     ('cs',): 'Czech',
     ('da',): 'Danish',
     ('de',): 'German',
     ('el',): 'Greek, Modern',
     ('en',): 'English',
     ('es',): 'Spanish',
     ('et',): 'Estonian',
     ('fi',): 'Finnish',
     ('fr',): 'French',
     ('hu',): 'Hungarian',
     ('it',): 'Italian',
     ('lt',): 'Lithuanian',
     ('lv',): 'Latvian',
     ('nl',): 'Dutch',
     ('pt',): 'Portuguese',
     ('ro',): 'Romanian',
     ('sk',): 'Slovak',
     ('sl',): 'Slovenian',
     ('sv',): 'Swedish'}

x = df.value_counts().index

print("The entered language is: ",l[x[0]])


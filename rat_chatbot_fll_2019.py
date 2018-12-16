import nltk
import numpy as np
import random
import string

rat = open('space.txt', 'r', errors='ignore')

raw = rat.read()

raw = raw.lower()

nltk.download('punkt')
nltk.download('wordnet')

sent_token = nltk.sent_tokenize(raw)
word_token = nltk.word_tokenize(raw)

lemmer = nltk.stem.WordNetLemmatizer()


def lemmer_tokens(tokens):
    return[lemmer.lemmatize(token) for token in tokens]


remove_punct = dict((ord(punct), None) for punct in string.punctuation)


def lemmer_normalize(text):
    return lemmer_tokens(nltk.word_tokenize(text.lower().translate(remove_punct)))


"""

<b>GREETING_INPUT</b> you can write as many as you can imagine that your user might write.

<b>GREETING_RESPONSE</b> is what your chatbot want you to say.

"""

GREETING_INPUT = ('hi', 'hello', 'hey', 'what\'s up dude!', 'greetings my friend')

GREETING_RESPONSE = ['Are you talking to me?', 'hey dude, nice to hear you', 'Yo my friend!']


def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUT:
            return random.choice(GREETING_RESPONSE)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def response(user_response):
    rat_response = ''

    TfidfVec = TfidfVectorizer(tokenizer=lemmer_normalize, stop_words='english')

    tfidf = TfidfVec.fit_transform(sent_token)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if req_tfidf == 0:
        rat_response = rat_response + "Understand you, I can't! Write clearer you must!"
        return rat_response
    else:
        rat_response = rat_response + sent_token[idx]
        return rat_response


"""

All the above sells will run and you'll see their number next to them.

This cell will have an asterisk * until you say bye.

"""


flag = True

print("RAT: My name is RAT. It means Robotic Academy of Thessaloniki.\
I will answer your queries about Space. If you want to exit, type Bye!")
while flag is True:
    user_response = input()
    user_response = user_response.lower()
    if user_response != 'bye':
        if user_response == 'thanks' or user_response == 'thank you':
            flag = False
            print("RAT: You are welcome..")
        else:
            if greeting(user_response) is not None:
                print("RAT: " + greeting(user_response))
            else:
                sent_token.append(user_response)
                word_token = word_token + nltk.word_tokenize(user_response)
                final_words = list(set(word_token))
                print("RAT: ", end="")
                print(response(user_response))
                sent_token.remove(user_response)
    else:
        flag = False
        print("RAT: Bye! take care..")
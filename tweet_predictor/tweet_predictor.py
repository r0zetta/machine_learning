from twarc import Twarc
import io, json, sys, os, random, re
import torch
from torch.nn import functional as F
from pytorch_pretrained_bert import GPT2Tokenizer, GPT2LMHeadModel

def load_json(fn):
    ret = None
    with io.open(fn, "r", encoding="utf-8") as f:
        ret = json.load(f)
    return ret

def preprocess_text(text):
    valid = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ#@'()-$,.!?/%\"\' "
    url_match = u"(https?:\/\/[0-9a-zA-Z\-\_]+\.[\-\_0-9a-zA-Z]+\.?[0-9a-zA-Z\-\_]*\/?.*)"
    name_match = u"\@[\_0-9a-zA-Z]+\:?"
    text = re.sub(url_match, u"", text)
    #text = re.sub(name_match, u"", text)
    text = re.sub(u"\&amp\;?", u"", text)
    text = re.sub(u"[\:\.]{1,}$", u"", text)
    text = re.sub(u"^RT\:?", u"", text)
    text = re.sub(u"/", u" ", text)
    text = re.sub(u"-", u" ", text)
    text = re.sub(u"\w*[\…]", u"", text)
    text = u''.join(x for x in text if x in valid)
    text = text.strip()
    return text

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

consumer_key="[redacted]"
consumer_secret="[redacted]"
access_token="[redacted]"
access_token_secret="[redacted]"

twarc = Twarc(consumer_key, consumer_secret, access_token, access_token_secret)

target_list = []
if (len(sys.argv) > 1):
    target_list = sys.argv[1:]
else:
    target_list.append("trump")

fh = open("output.txt", "a")

query = ",".join(target_list)
print("Search: " + query)
for status in twarc.filter(track = query):
    tweet = status["text"].replace("\n", " ")
    if "…" not in tweet:
        continue
    if random.random() > 0.90:
        start = preprocess_text(tweet)
        if len(start) < 70:
            continue
        tweet_url = "https://twitter.com/" + status["user"]["screen_name"] + "/status/" + status["id_str"]
        msg = "\n"
        msg += "=================\n"
        msg += tweet_url + "\n"
        msg += "Input: " + tweet + "\n"
        msg += "=================\n"
        vecs = tokenizer.encode(start)
        inp, past = torch.tensor([vecs]), None
        for _ in range(40):
            logits, past = model(inp, past=past)
            inp = torch.multinomial(F.softmax(logits[:, -1]), 1)
            vecs.append(inp.item())
        text = tokenizer.decode(vecs).replace("\n", " ")
        msg += "Output: " + text + "\n"
        msg += "=================\n"
        msg += "\n"
        print(msg)
        fh.write(msg)

import io, json, sys, os, re
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

tweets_raw = []
with io.open("trump_tweets.txt") as f:
    for line in f:
        tweets_raw.append(line)

random.shuffle(tweets_raw)

fh = open("trump_predictions.txt", "a")

for tweet in tweets_raw:
    if "…" not in tweet:
        continue
    cleaned = preprocess_text(tweet)
    if cleaned is None or len(cleaned) < 20:
        continue
    msg = "\n"
    msg += "=================\n"
    msg += "Original: " + tweet.replace("\n", " ") + "\n"
    msg += "=================\n"
    outputs = []
    for _ in range(5):
        vecs = tokenizer.encode(cleaned)
        inp, past = torch.tensor([vecs]), None
        words = 0
        end_sent = False
        while end_sent == False:
            logits, past = model(inp, past=past)
            inp = torch.multinomial(F.softmax(logits[:, -1]), 1)
            vecs.append(inp.item())
            words += 1
            if words > 15:
                temp = tokenizer.decode(vecs)
                if temp[-1] == ".":
                    end_sent = True
            if words > 40:
                end_sent = True
        text = tokenizer.decode(vecs)
        text = text.replace("\n", " ")
        outputs.append(text)
    for n, t in enumerate(outputs):
        msg += "Output " + str(n+1) + ": " + t + "\n"
    msg += "=================\n"
    msg += "\n"
    print(msg)
    fh.write(msg)

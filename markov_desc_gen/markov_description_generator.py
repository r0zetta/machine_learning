import markovify, os, time

# Load descriptions
print("Loading data")
descs = set()
with open("descs.txt", "r") as f:
    for line in f:
        descs.add(line.strip())
print("Done")

# Train model
print("Preparing model")
markov_model = markovify.Text(descs)
print("Done")

print("")
while True:
    sent = markov_model.make_sentence()
    if sent is not None and len(sent) > 140 and len(sent) < 300:
        words = sent.split()
        new_sent = "  "
        cur_len = 0
        sent_h = 1
        good_sent = True
        for w in words:
            wlen = len(w)
            if wlen > 30:
                good_sent = False
            if cur_len + wlen > 30:
                new_sent += "\n  "
                sent_h += 1
                cur_len = 0
            new_sent += " " + w
            cur_len += wlen
        if good_sent == True:
            os.system("clear")
            print("")
            print("")
            print(new_sent)
            print("")
            print("")
            time.sleep(3)

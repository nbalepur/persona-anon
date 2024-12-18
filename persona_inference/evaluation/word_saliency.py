
dataset = 'BeaverTails' # dataset name
models = ['gpt-4o-mini'] # models with personas to test
run_name = 'default' # run name
dataset_name = 'nbalepur/persona-inference' # huggingface dataset name

FREQUENCY_CUTOFF = 3 # number of times words must appear to be valid

import datasets
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
lemmatizer = WordNetLemmatizer()
# ====
# nltk.download('punkt_tab')
# ==== > run this the first time

ds = datasets.load_dataset(dataset_name, dataset)['train']
chosen_personas = []
rejected_personas = []

for model in models:

    f = f'results/{model}/{dataset}/{run_name}/persona_inference.jsonl'
    with open(f, 'r') as json_file:
        json_list = list(json_file)

    assert len(json_list) == (2 * ds.num_rows), "Your number of personas doesn't match the size of the dataset (did you run persona inference on this full dataset?)"

    chosen_personas.extend(json_list[:ds.num_rows])
    rejected_personas.extend(json_list[ds.num_rows:])


all_personas = chosen_personas + rejected_personas
all_labels = ['chosen' for _ in chosen_personas] + ['rejected' for _ in rejected_personas]

label_ctr = dict()
word_ctr = dict()

split_words = ['rather than', 'over', 'versus', 'compared to', 'instead of']

for p, l in zip(all_personas, all_labels):
    persona = nltk.sent_tokenize(p.lower())[0]
    for sw in split_words:
        persona = persona.split(sw)[0]
    for w in nltk.word_tokenize(persona):
        w_ = w.lower().replace(',', '').replace('.', '').replace("'", '')
        w = lemmatizer.lemmatize(w_)
        label_ctr[w, l] = label_ctr.get((w, l), 0) + 1
        word_ctr[w] = word_ctr.get(w, 0) + 1

p_label_given_word = dict()
for w, ct in word_ctr.items():
    if ct < FREQUENCY_CUTOFF:
        continue
    for l in ['chosen', 'rejected']:
        p_label_given_word[l, w] = (1.0 * label_ctr.get((w, l), 0)) / word_ctr[w]
    
p_label_given_word_chosen = [item for item in p_label_given_word.items() if item[0][0] == 'chosen']
p_label_given_word_rejected = [item for item in p_label_given_word.items() if item[0][0] == 'rejected']

p_label_given_word_chosen = sorted(p_label_given_word_chosen, key=lambda item: -1 * item[1])
p_label_given_word_rejected = sorted(p_label_given_word_rejected, key=lambda item: -1 * item[1])

print("Top-10 Most Salient Chosen Persona Words")
print(p_label_given_word_chosen[:10])
print()
print("Top 10 Most Salient Rejected Persona Words")
print(p_label_given_word_rejected[:10])
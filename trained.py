from __future__ import unicode_literals, print_function

import random
from pathlib import Path
import spacy
from tqdm import tqdm
from spacy.training import Example
from data import PreprocessData
import os
import ast


class CustomNer:

    def __init__(self, text, model, pretrained_model, trained_model):

        self.model = model
        self.text = text
        self.pretrained_model = pretrained_model
        self.trained_model = trained_model

    def check_model(self):

        if not self.model and not self.trained_model:
            nlp = spacy.load(self.model)
            print("Loaded model '%s'" % self.model)
        elif self.pretrained_model:
            nlp = spacy.load('en_core_web_lg')
        else:
            nlp = spacy.blank('en')
            print("Created blank 'en' model")
        return nlp

    def set_pipelines(self):
        nlp = self.check_model()
        if 'ner' not in nlp.pipe_names:
            ner = nlp.create_pipe('ner')
            nlp.add_pipe('ner', last=True)
        else:
            ner = nlp.get_pipe('ner')

        return ner, nlp

    def train_model(self, n_iter=100):

        ner, nlp = self.set_pipelines()
        print(f'nlp.pipe_names {nlp.pipe_names}')
        print(f'ner {ner}')
        data_init = PreprocessData(self.text)
        data = data_init.process_data()
        for _, annotations in data:
            for enti in annotations.get('entities'):
                ner.add_label(enti[2])
        print(ner)
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        with nlp.disable_pipes(*other_pipes):  # only train NER
            print(f'self.model : {self.model}')
            if self.pretrained_model and self.trained_model:

                print(f'check for nlp {nlp.pipe_names}')
                optimizer = nlp.create_optimizer()
            else:
                print(f'check for nlp: This is  {nlp.pipe_names}')
                optimizer = nlp.begin_training()
            for itn in range(n_iter):
                random.shuffle(data)
                losses = {}
                for text, annotations in tqdm(data):
                    doc_text = nlp.make_doc(text)
                    example = Example.from_dict(doc_text, annotations)
                    nlp.update(
                        [example],
                        drop=0.5,
                        sgd=optimizer,
                        losses=losses)
                #print(losses)
        return nlp


def save_model(out_dir, nlp):
    if out_dir is not None:
        output_dir = Path(out_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)


def open_model(out_dir):
    print("Loading from", out_dir)
    path = os.path.join(out_dir, 'meta.json')
    file_exist = os.path.isfile(path)
    print(f'path exist {file_exist}')
    if file_exist:
        nlp_mod = spacy.load(out_dir)
    else:
        nlp_mod = None

    return nlp_mod


def test_model(test_text, nlp_model_trained):
    print(f'nlp_nodel : {nlp_model_trained}')
    doc = nlp_model_trained(test_text)
    for ent in doc.ents:
        print(ent)
    print('Entities', [(ent.text, ent.label_) for ent in doc.ents])


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))  # <-- absolute dir the script is in
    rel_path = "data.txt"
    abs_file_path = os.path.join(script_dir, rel_path)
    model_path = 'model'
    abs_model_path = os.path.join(script_dir, model_path)
    model_nlp = open_model(abs_model_path)
    # declare if you want to use already trained spacy model or not
    pretrained_without_spacy = 'spacy'
    spacy_train = 'train'
    if model_nlp and not spacy_train:
        test = 'my account id is 564364 and acci no 12/24/2020 ddcdfvd account number \
                 and the values acct id has a bill date'
        # test = 'account number is for'
        test_model(test, model_nlp)
    elif spacy_train:
        with open('train_data', 'r') as f:
            ds_string = ast.literal_eval(f.read())
            model_nlp = open_model(abs_model_path)
            x = CustomNer(ds_string, model_nlp, pretrained_without_spacy, spacy_train)
            nlp_model = x.train_model()
            save_model(abs_model_path, nlp_model)

            test = 'my account id is 564364 and acci no 12/24/2020 ddcdfvd account number \
             and the values acct id has a bill date form the used date until the account no is dead'
            # test = 'account number is for'
            test_model(test, nlp_model)



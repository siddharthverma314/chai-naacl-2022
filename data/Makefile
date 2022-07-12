# Makefile to download all data
URL = curl $1 > $@

## All
.PHONY: all
all: dataset baselines

## datasets
schema.json:
	$(call URL,https://worksheets.codalab.org/rest/bundles/0xd5a278f7595e4ca2aec65fd8bef52c08/contents/blob/)

train.json:
	$(call URL,https://worksheets.codalab.org/rest/bundles/0xda2bae7241044dbaa4e8ebb02c280d8f/contents/blob/)

dev.json:
	$(call URL,curl https://worksheets.codalab.org/rest/bundles/0xb0fe71ca124e43f6a783324734918d2c/contents/blob/)

test.json:
	$(call URL,https://worksheets.codalab.org/rest/bundles/0x54d325bbcfb2463583995725ed8ca42b/contents/blob/)

parsed.json:
	$(call URL,https://worksheets.codalab.org/rest/bundles/0xd34bbbc5fb3b4fccbd19e10756ca8dd7/contents/blob/parsed.json)

.PHONY: dataset
dataset: schema.json train.json dev.json test.json parsed.json

## Baselines
model_utility.pt:
	$(call URL,https://worksheets.codalab.org/rest/bundles/0xd658de343912461598ce53dc9354dc60/contents/blob/checkpoint/lf2lf-margin/model_best.pt)

model_length.pt:
	$(call URL,https://worksheets.codalab.org/rest/bundles/0x5120cbb5102e41058c66a80fc34107d1/contents/blob/checkpoint/lf2lf-margin/model_best.pt)

model_fairness.pt:
	$(call URL,https://worksheets.codalab.org/rest/bundles/0x4d43dc62b06c4dcab6628c16f20de0a4/contents/blob/checkpoint/lf2lf-margin/model_best.pt)

model_supervised.pt:
	$(call URL,https://worksheets.codalab.org/rest/bundles/0xab2055ab75de4c9c825a804795ddb120/contents/blob/checkpoint/lf2lf/model_best.pt)

price_tracker.pkl:
	$(call URL,https://worksheets.codalab.org/rest/bundles/0x91bdbaf3fc4a43afb900584ac73a88b3/contents/blob/price_tracker.pkl)

vocab.pkl:
	$(call URL,https://worksheets.codalab.org/rest/bundles/0xab2055ab75de4c9c825a804795ddb120/contents/blob/mappings/lf2lf/vocab.pkl)

templates.pkl:
	$(call URL,https://worksheets.codalab.org/rest/bundles/0xd34bbbc5fb3b4fccbd19e10756ca8dd7/contents/blob/templates.pkl)

.PHONY: baselines
baselines: model_utility.pt model_length.pt model_fairness.pt model_supervised.pt price_tracker.pkl vocab.pkl templates.pkl

## Clean
.PHONY: clean
clean:
	-@cat .gitignore | perl -nE "for(glob){say}" | xargs rm -rv

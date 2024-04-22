
OPTIONS ?=

.PHONY: \
	all \
	clean \
	experiment1 \
	stability_test


all: stability_test

clean:
	rm -R data_rec
	rm experiments/1/*.png
	rm experiments/stability_test/*.png

experiment1 :
	python experiments/1/001-greedy.py $(OPTIONS)

stability_test: experiment1
	python experiments/stability_test/model_50.py $(OPTIONS)


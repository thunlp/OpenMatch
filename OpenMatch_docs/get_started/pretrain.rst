Pretrain
==========

We now have the following released models:

* **wiki80_cnn_softmax**: trained on **wiki80** dataset with a CNN encoder.
* **wiki80_bert_softmax**: trained on **wiki80** dataset with a BERT encoder.

How to Use Them
------------------

After installing OpenNRE, open Python and load OpenNRE:

::
  
  >>> import opennre
  
Then load the model with its corresponding name:

::
  
  >>> model = opennre.get_model('wiki80_cnn_softmax')
  
  
It may take a little while to download the model. After loading it, you can do relation extraction with the following format:

::
  
  >>> model.infer({'text': 'He was the son of Máel Dúin mac Máele Fithrich, and grandson of the high king Áed Uaridnach (died 612).', 'h': {'pos': (18, 46)}, 't': {'pos': (78, 91)}})

The ``infer`` function takes one dict as input. The `text` key represents the sentence and `h` / `t` keys represent head and tail entities, in which `pos` (position) should be specified.

The model will return the predicted result as:

::

  ('father', 0.5108704566955566)

User Guide
====

Use our pre-trained models
---------------------------

You can use our pre-trained models:

* **wiki80_cnn_softmax**: trained on **wiki80** dataset with a CNN encoder.
* **wiki80_bert_softmax**: trained on **wiki80** dataset with a BERT encoder.

In the following way:

1. Load the model

::
  
  >>> import opennre
  >>> model = opennre.get_model('wiki80_cnn_softmax')

2. Use ``infer`` to do relation extraction
  
::
  
  >>> model.infer({'text': 'He was the son of Máel Dúin mac Máele Fithrich, and grandson of the high king Áed Uaridnach (died 612).', 'h': {'pos': (18, 46)}, 't': {'pos': (78, 91)}})

The model will return the predicted result as:

::

  ('father', 0.5108704566955566)


Train your own models
----------------------

We have examples to train relation extraction models in the ``example`` folder:

* ``train_supervised_cnn.py``: train a supervised CNN model.
* ``train_supervised_bert.py``: train a supervised BERT model.
* ``train_bag_pcnn_att.py``: train a bag-level PCNN-ATT model.

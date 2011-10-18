# -*- coding: utf-8 -*-
# Natural Language Toolkit: Interface to the Stanford NER-tagger
#
# Copyright (C) 2001-2011 NLTK Project
# Author: Nitin Madnani <nmadnani@ets.org>
# URL: <http://www.nltk.org/>
# For license information, see LICENSE.TXT
#
# $Id: stanford.py $

"""
A module for interfacing with the Stanford taggers.
"""

import os
from subprocess import PIPE
import tempfile
import nltk
from api import *

_stanford_url = 'http://nlp.stanford.edu/software'

class StanfordTagger(TaggerI):
    """
    An interface to Stanford taggers. Subclasses must define:
      - L{_cmd} property: A property that returns the command that will be
        executed.
      - _SEPRARTOR: Class constant that represents that character that is used
        to separate the tokens from their tags.
      - _JAR file: Class constant that represents the jar file name.
    """

    _SEPARATOR = ''
    _JAR = ''

    def __init__(self, path_to_model, path_to_jar=None, encoding=None, verbose=False, java_options='-mx1000m'):

        self._stanford_jar = nltk.internals.find_jar(
                self._JAR, path_to_jar,
                searchpath=(), url=_stanford_url,
                verbose=verbose)

        if not os.path.isfile(path_to_model):
            raise IOError("Stanford tagger model file not found: %s" % path_to_model)
        self._stanford_model = path_to_model
        self._encoding = encoding
        self.java_options = java_options

    @property
    def _cmd(self):
      raise NotImplementedError

    def tag(self, tokens):
        return self.batch_tag([tokens])[0]

    def batch_tag(self, sentences):
        encoding = self._encoding
        default_options = ' '.join(nltk.internals._java_options)
        nltk.internals.config_java(options=self.java_options, verbose=False)

        # Create a temporary input file
        _input_fh, self._input_file_path = tempfile.mkstemp(text=True)

        if encoding:
            self._cmd.extend(['-encoding', encoding])

        # Write the actual sentences to the temporary input file
        _input_fh = os.fdopen(_input_fh, 'w')
        _input = '\n'.join((' '.join(x) for x in sentences))
        if isinstance(_input, unicode) and encoding:
            _input = _input.encode(encoding)
        _input_fh.write(_input)
        _input_fh.close()

        # Run the tagger and get the output
        stanpos_output, _stderr = nltk.internals.java(self._cmd,classpath=self._stanford_jar, \
                                                       stdout=PIPE, stderr=PIPE)
        if encoding:
            stanpos_output = stanpos_output.decode(encoding)

        # Delete the temporary file
        os.unlink(self._input_file_path)

        # Return java configurations to their default values
        nltk.internals.config_java(options=default_options, verbose=False)

        # Output the tagged sentences
        tagged_sentences = []
        for tagged_sentence in stanpos_output.strip().split("\n"):
            sentence = [tuple(tagged_word.strip().split(self._SEPARATOR))
                        for tagged_word in tagged_sentence.strip().split()]
            tagged_sentences.append(sentence)
        return tagged_sentences

class POSTagger(StanfordTagger):
    """
    A class for pos tagging with Stanford Tagger. The input is the paths to:
     - a model trained on training data
     - (optionally) the path to the stanford tagger jar file. If not specified here,
       then this jar file must be specified in the CLASSPATH envinroment variable.
     - (optionally) the encoding of the training data (default: ASCII)

    Example:

        >>> st = POSTagger('bidirectional-distsim-wsj-0-18.tagger')
        >>> st.tag('What is the airspeed of an unladen swallow ?'.split())
        [('What', 'WP'), ('is', 'VBZ'), ('the', 'DT'), ('airspeed', 'NN'), ('of', 'IN'), ('an', 'DT'), ('unladen', 'JJ'), ('swallow', 'VB'), ('?', '.')]
    """

    _SEPARATOR = '_'
    _JAR = 'stanford-postagger.jar'

    def __init__(self, *args, **kwargs):
        super(POSTagger, self).__init__(*args, **kwargs)
    
    @property
    def _cmd(self):
        return ['edu.stanford.nlp.tagger.maxent.MaxentTagger', \
                '-model', self._stanford_model, '-textFile', \
                self._input_file_path, '-tokenize', 'false']

class NERTagger(StanfordTagger):
    """
    A class for ner tagging with Stanford Tagger. The input is the paths to:
     - a model trained on training data
     - (optionally) the path to the stanford tagger jar file. If not specified here,
       then this jar file must be specified in the CLASSPATH envinroment variable.
     - (optionally) the encoding of the training data (default: ASCII)

    Example:

        >>> st = NERTagger('all.3class.distsim.crf.ser.gz')
        >>> st.tag('Rami Eid is studying at Stony Brook University in NY'.split())
        [('Rami', 'PERSON'), ('Eid', 'PERSON'), ('is', 'O'), ('studying', 'O'),
         ('at', 'O'), ('Stony', 'ORGANIZATION'), ('Brook', 'ORGANIZATION'),
         ('University', 'ORGANIZATION'), ('in', 'O'), ('NY', 'LOCATION')]
    """

    _SEPARATOR = '/'
    _JAR = 'stanford-ner.jar'

    def __init__(self, *args, **kwargs):
        super(NERTagger, self).__init__(*args, **kwargs)

    @property
    def _cmd(self):
        return ['edu.stanford.nlp.ie.crf.CRFClassifier', \
                '-loadClassifier', self._stanford_model, '-textFile', \
                self._input_file_path, '-outputFormat', 'slashTags']

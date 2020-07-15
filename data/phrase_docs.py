import data.preprocessor as preprocessor
import ansaro_utils.data.text_utils as au_tu

from pathlib import Path
import sys
import os

import pandas as pd

import spacy
import codecs
import re
import itertools as it  # for printing examples
from collections import OrderedDict

from gensim.models.word2vec import LineSentence
from gensim.models import Phrases
from gensim.models.phrases import Phraser

from smart_open import smart_open


import logging
#logging.basicConfig(format='{asctime} : {levelname} : {message}', level=logging.INFO, style='{')
logger = logging.getLogger(__name__)

# add this module to runtime path
sys.path.append(Path.cwd().parent.as_posix())


to_load = 'text'
verbose = True

LOCAL_DATA = Path(os.environ['LOCAL_DATA'])
data_dir_local = LOCAL_DATA / 'topic_modeling'
# data_dir_local = REPO_ROOT / 'data'

data_dir = data_dir_local
# data_dir_raw = data_dir / 'raw'
data_dir_interim = data_dir / 'interim' / 'topic_modeling'
data_dir_training_processed = data_dir / 'training'/'processed'
data_dir_inference_processed = data_dir / 'inference'/'processed'

if not data_dir_interim.exists():
    data_dir_interim.mkdir(parents=True, exist_ok=True)
if not data_dir_training_processed.exists():
    data_dir_training_processed.mkdir(parents=True, exist_ok=True)
if not data_dir_inference_processed.exists():
    data_dir_inference_processed.mkdir(parents=True, exist_ok=True)

whole_words_only = True
ignore_case = True

# tokens to clean, for consistency across documents
US_STR = 'United_States'
TFA_STR = 'Teach_For_America'
CORPS_MEMBER_STR = 'Corps_Member'
AMERICORPS_STR = 'AmeriCorps'
MRS_W_STR = 'Mrs_Walker'
DREW_STR = 'Drew_Student'
# KEN_STR = 'Kennedy_Student'
LOW_INCOME_STR = 'low_income'
UC_STR = 'upper_class'
MC_STR = 'middle_class'
LC_STR = 'lower_class'
LI_STR = 'low_income'
IC_STR = 'inner_city'
HS_STR = 'high_school'
HSS_STR = 'high_school_student'
ZIP_STR = 'zip_code'
AC_STR = 'area_code'
SES_STR = 'socioeconomic_status'
SE_STR = 'socioeconomic'
to_replace_dict = OrderedDict({
    'United States of America': US_STR,
    'United States': US_STR,
    'U. S. A': US_STR,
    'U.S.A': US_STR,
    'USA': US_STR,
    'U.S': US_STR,
    'U. S': US_STR,
    'US': US_STR,
    'Teaching For America': TFA_STR,
    'Teach For America': TFA_STR,
    'Teach Fro America': TFA_STR,
    'TeachForAmerica': TFA_STR,
    'TeachFor America': TFA_STR,
    'Teach ForAmerica': TFA_STR,
    'TFA': TFA_STR,
    'T4A': TFA_STR,
    'Teach Of America': TFA_STR,
    'corps members': CORPS_MEMBER_STR,
    'corp members': CORPS_MEMBER_STR,
    'corpsmembers': CORPS_MEMBER_STR,
    'corpmembers': CORPS_MEMBER_STR,
    'corps member': CORPS_MEMBER_STR,
    'corp member': CORPS_MEMBER_STR,
    'corpsmember': CORPS_MEMBER_STR,
    'corpmember': CORPS_MEMBER_STR,
    'core members': CORPS_MEMBER_STR,
    'core members': CORPS_MEMBER_STR,
    'coremembers': CORPS_MEMBER_STR,
    'core member': CORPS_MEMBER_STR,
    'core member': CORPS_MEMBER_STR,
    'coremember': CORPS_MEMBER_STR,
    'AmeriCorps': AMERICORPS_STR,
    'Ameri Corps': AMERICORPS_STR,
    'AmeriCorp': AMERICORPS_STR,
    'Ameri Corp': AMERICORPS_STR,
    'Ameri Core': AMERICORPS_STR,
    'AmeriCore': AMERICORPS_STR,
    'low-income': LOW_INCOME_STR,
    'low income': LOW_INCOME_STR,
    'inner city': IC_STR,
    'inner-city': IC_STR,
    'innercity': IC_STR,
    'upper class': UC_STR,
    'upper-class': UC_STR,
    'upperclass': UC_STR,
    'middle class': MC_STR,
    'middle-class': MC_STR,
    'middleclass': MC_STR,
    'lower class': LC_STR,
    'lower-class': LC_STR,
    'lowerclass': LC_STR,
    'low income': LI_STR,
    'low-income': LI_STR,
    'lowincome': LI_STR,
    'lower income': LI_STR,
    'lower-income': LI_STR,
    'lowerincome': LI_STR,
    'high schools': HS_STR,
    'highschools': HS_STR,
    'high school': HS_STR,
    'highschool': HS_STR,
    'high school students': HSS_STR,
    'high school student': HSS_STR,
    'high schoolers': HSS_STR,
    'highschoolers': HSS_STR,
    'high schooler': HSS_STR,
    'highschooler': HSS_STR,
    'zip codes': ZIP_STR,
    'zipcodes': ZIP_STR,
    'zip code': ZIP_STR,
    'zipcode': ZIP_STR,
    'area codes': AC_STR,
    'areacodes': AC_STR,
    'area code': AC_STR,
    'areacode': AC_STR,
    'Mrs. Walker': MRS_W_STR,
    'Mrs Walker': MRS_W_STR,
    'Ms. Walker': MRS_W_STR,
    'Ms Walker': MRS_W_STR,
    'Drew': DREW_STR,
    'Andrew': DREW_STR,
    # 'Kennedy': KEN_STR,
    # 'Kenedy': KEN_STR,
    # 'Kenneddy': KEN_STR,
    # 'Keneddy': KEN_STR,
    'socio economic status': SES_STR,
    'socio-economic-status': SES_STR,
    'socio-economic status': SES_STR,
    'socioeconomic-status': SES_STR,
    'socioeconomic status': SES_STR,
    'SES': SES_STR,
    'socio-economics': SE_STR,
    'socio economics': SE_STR,
    'socio-economic': SE_STR,
    'socio economic': SE_STR,
    'medicaid': 'Medicaid',
    'medicare': 'Medicare',
    'co-workers': 'co_worker',
    'coworkers': 'co_worker',
    'co-worker': 'co_worker',
    'coworker': 'co_worker',
    '1st': 'first',
    '2nd': 'second',
    '3rd': 'third',
    '4th': 'fourth',
    '5th': 'fifth',
    '6th': 'sixth',
    '7th': 'seventh',
    '8th': 'eighth',
    '9th': 'ninth',
    '10th': 'tenth',
    '11th': 'eleventh',
    '12th': 'twelfth',
    'role model': 'role_model',
    'communities': 'community',
    'neighborhoods': 'community',  # maybe?
    'neighborhood': 'community',  # maybe?
    # 'I know': 'I_know',
    # 'I want': 'I_want',
    # 'I believe': 'I_believe',
    'work hard': 'work_hard',
    'hard work': 'hard_work',
    'special needs': 'special_needs',
    'behavior management': 'behavior_management',
    'classroom management': 'classroom_management',
})

to_remove_list = []

nlp_batch_size = 1000
# nlp_n_threads = 2
nlp_n_threads = -1
overwrite_interim = True

phrase_max_vocab_size = 40000000
phrase_min_count = 5
# phrase_min_count = 2  # debug
phrase_threshold = 60  # was 500
phrase_progress_per = 500
phrase_scoring = 'default'
phrase_common_terms = frozenset(['of', 'with', 'without',
                                 'and', 'or', 'the', 'a',
                                 'not', 'be', 'to', 'this',
                                 'who', 'in', 'for',
                                 ])

response_cols = ['resp_whytfa']


def raw_to_phrased_data_pipeline(to_load: str, verbose=True, overwrite_interim=True, prefix=None, training=True) -> None:
    if prefix is None:
        prefix = ''
    df, cols_dict = preprocessor.load_processed_data(to_load=to_load, verbose=verbose, training=training)
    df = preprocessor.response_len(df)
    if training is True:
        df = preprocessor.remove_shortest_essays(df, thresh=0.0025)

    # debug
    for col in response_cols:
        # for nlp_preprocess
        filename_phrased = f'{prefix}{col}_phrased.csv'

        # for saving locally
        doc_txt_filepath = data_dir_interim / f'{prefix}{col}_doc_text_all.txt'
        unigram_sentences_filepath = data_dir_interim / f'{prefix}{col}_unigram_sentences_all.txt'
        bigram_model_filepath = data_dir_interim / f'{prefix}{col}_bigram_model_all'
        bigram_sentences_filepath = data_dir_interim / f'{prefix}{col}_bigram_sentences_all.txt'
        trigram_model_filepath = data_dir_interim / f'{prefix}{col}_trigram_model_all'
        trigram_sentences_filepath = data_dir_interim / f'{prefix}{col}_trigram_sentences_all.txt'
        if training is True:
            trigram_docs_filepath = data_dir_training_processed / f'{prefix}{col}_trigram_transformed_docs_all.txt'
        else:
            trigram_docs_filepath = data_dir_inference_processed / f'{prefix}{col}_trigram_transformed_docs_all.txt'

        # turn to posix filepaths until gensim supports this
        filepath_dict = {}
        filepath_dict['filepath_in'] = None
        if training is True:
            filepath_dict['filepath_out'] = data_dir_training_processed / filename_phrased
        else:
            filepath_dict['filepath_out'] = data_dir_inference_processed / filename_phrased
        filepath_dict['doc_txt_filepath'] = doc_txt_filepath.as_posix()
        filepath_dict['unigram_sentences_filepath'] = unigram_sentences_filepath.as_posix()
        filepath_dict['bigram_model_filepath'] = bigram_model_filepath.as_posix()
        filepath_dict['bigram_sentences_filepath'] = bigram_sentences_filepath.as_posix()
        filepath_dict['trigram_model_filepath'] = trigram_model_filepath.as_posix()
        filepath_dict['trigram_sentences_filepath'] = trigram_sentences_filepath.as_posix()
        filepath_dict['trigram_docs_filepath'] = trigram_docs_filepath.as_posix()

        return nlp_preprocess(filepath_dict,
                                    col=col, df=df, verbose=verbose,
                                    overwrite_interim=overwrite_interim)


def nlp_preprocess(filepath_dict: dict, col: str,
                   df=None, verbose: bool = True,
                   overwrite_interim: bool = True) -> pd.DataFrame:
    def clean_doc(corpus):
        '''
        generator function to read in docs from the file,
        and substitute and remove substrings
        '''
        for doc in corpus:
            yield au_tu.remove_substrings(
                au_tu.clean_tokens(
                    doc, tokens=to_replace_dict,
                    whole_words_only=whole_words_only,
                    ignore_case=ignore_case,),
                to_remove_list=to_remove_list,
                whole_words_only=whole_words_only,
                ignore_case=ignore_case)

    def tokenize_entities(parsed_doc):
        txt = parsed_doc.text
        for ent in parsed_doc.ents:
            txt = txt[:ent.start_char] + ent.text.replace(' ', '_') + txt[ent.end_char:]
        return txt

    def cleaned_doc_corpus(corpus):
        '''
        generator function to use spaCy to parse docs, clean docs,
        tokenize named entities, and yield documents
        '''
        for parsed_doc in nlp.pipe(clean_doc(corpus),
                                   batch_size=nlp_batch_size,
                                   n_threads=nlp_n_threads):
            yield tokenize_entities(parsed_doc)

    def punct_space_more(token):
        '''
        helper function to eliminate tokens that are
        pure punctuation or whitespace or digits or only 1 character
        '''
        return (token.is_punct or
                token.is_space or
                token.is_digit or
                token.text == "'s" or
                token.lemma_ == '-PRON-' or
                # token.lemma_ == 'say' or
                # token.lemma_ == 'tell' or
                # token.lemma_ == 'be' or
                len(token.text) <= 1)

    def line_doc(filename):
        '''
        generator function to read in docs from the file,
        un-escape the original line breaks in the text,
        and do additional cleaning
        '''

        def hyp_to_us(doc):
            return re.sub(r'\b-\b', '_', doc)

        def remove_punct(doc):
            # keep: alphanumberic (w), spaces (s), single quote, underscore
            return re.sub(r'[^\w\s\'_]+', '', doc)

        # with codecs.open(filename, encoding='utf_8') as f:
        with smart_open(filename) as f:
            for doc in f:
                yield remove_punct(
                    hyp_to_us(doc.decode())
                ).replace('\\n', '\n')

    def lemmatized_sentence_corpus(filename):
        '''
        generator function to use spaCy to parse docs,
        lemmatize the text, and yield sentences
        '''
        for parsed_doc in nlp.pipe(line_doc(filename),
                                   batch_size=nlp_batch_size,
                                   n_threads=nlp_n_threads):

            for sent in parsed_doc.sents:
                yield ' '.join([token.lemma_ for token in sent
                                if not punct_space_more(token)])

    if verbose:
        logger.info(f'Working on text from: {col}')

    # # debug - only getting from the sample dataframe here
    # df_phrased = df.loc[df[col].notnull(), ['tfa_master_uid', 'app_year', col]].sample(n=50).copy()

    df_phrased = df.loc[df[col].notnull(), ['tfa_master_uid', 'app_year', col]].copy()

    nlp = spacy.load('en', disable=[])

    # clean text and tokenize entities
    if verbose:
        logger.info('Cleaning docs...')
    df_phrased[col] = list(cleaned_doc_corpus(df_phrased[col].values))
    # remove 'the_' from NER tokens
    df_phrased[col] = df_phrased[col].apply(lambda x: ' '.join([re.sub('^the_', 'the ', y) for y in x.split()]))
    if verbose:
        logger.info('\tDone.')

    # create & open a new file in write mode
    if verbose:
        logger.info('Saving documents, one per line...')
    doc_count = 0
    with codecs.open(filepath_dict['doc_txt_filepath'], 'w', encoding='utf_8') as doc_txt_file:
        for doc in df_phrased[[col]].apply(lambda x: ' '.join(x), axis=1).tolist():
            # write the doc as a line in the new file
            # escape newline characters in the original doc text
            doc_txt_file.write(doc.replace('\n', '\\n') + '\n')
            doc_count += 1
    if verbose:
        logger.info(f"Text from {doc_count:,} docs written to: {filepath_dict['doc_txt_filepath']}")

    nlp = spacy.load('en', disable=['ner'])

    # lemmatize and save sentences

    if overwrite_interim:
        if verbose:
            logger.info(f"Processing documents into unigram sentences: {filepath_dict['unigram_sentences_filepath']}")
        # with codecs.open(filepath_dict['unigram_sentences_filepath'], 'w', encoding='utf_8') as f:
        with smart_open(filepath_dict['unigram_sentences_filepath'], 'w') as f:
            for sentence in lemmatized_sentence_corpus(filepath_dict['doc_txt_filepath']):
                f.write(sentence + '\n')
            if verbose:
                logger.info('Done.')
        unigram_sentences = LineSentence(filepath_dict['unigram_sentences_filepath'])

        if verbose:
            logger.info('Unigram examples:')
            for unigram_sentence in it.islice(unigram_sentences, 10, 20):
                logger.info(u' '.join(unigram_sentence))
                logger.info('=' * 30)

        if verbose:
            logger.info('Finding bigram phrases')
        # create the bigram model
        bigram = Phrases(unigram_sentences,
                         min_count=phrase_min_count,
                         threshold=phrase_threshold,
                         max_vocab_size=phrase_max_vocab_size,
                         progress_per=phrase_progress_per,
                         scoring=phrase_scoring,
                         common_terms=phrase_common_terms
                         )
        bigram_model = Phraser(bigram)
        bigram_model.save(filepath_dict['bigram_model_filepath'])

        if verbose:
            logger.info(f"Saving bigram phrased sentences: {filepath_dict['bigram_sentences_filepath']}")
        # save bigram sentences
        with codecs.open(filepath_dict['bigram_sentences_filepath'], 'w', encoding='utf_8') as f:
            for unigram_sentence in unigram_sentences:
                bigram_sentence = u' '.join(bigram_model[unigram_sentence])
                f.write(bigram_sentence + '\n')

        bigram_sentences = LineSentence(filepath_dict['bigram_sentences_filepath'])
        if verbose:
            logger.info('Bigram examples:')
            for bigram_sentence in it.islice(bigram_sentences, 10, 20):
                logger.info(u' '.join(bigram_sentence))
                logger.info('=' * 30)

        if verbose:
            logger.info('Finding trigram phrases')
        # create the trigram model
        trigram = Phrases(bigram_sentences,
                          min_count=phrase_min_count,
                          threshold=phrase_threshold,
                          max_vocab_size=phrase_max_vocab_size,
                          progress_per=phrase_progress_per,
                          scoring=phrase_scoring,
                          common_terms=phrase_common_terms
                          )
        trigram_model = Phraser(trigram)
        trigram_model.save(filepath_dict['trigram_model_filepath'])

        if verbose:
            logger.info(f"Saving trigram phrased sentences: {filepath_dict['trigram_sentences_filepath']}")
        # create trigram sentences
        with codecs.open(filepath_dict['trigram_sentences_filepath'], 'w', encoding='utf_8') as f:
            for bigram_sentence in bigram_sentences:
                trigram_sentence = u' '.join(trigram_model[bigram_sentence])
                f.write(trigram_sentence + '\n')

        trigram_sentences = LineSentence(filepath_dict['trigram_sentences_filepath'])
        if verbose:
            logger.info('Trigram examples:')
            for trigram_sentence in it.islice(trigram_sentences, 10, 20):
                logger.info(u' '.join(trigram_sentence))
                logger.info('=' * 30)

    if verbose:
        logger.info(f"Saving phrased docs using saved models: {filepath_dict['trigram_docs_filepath']}")
    # using saved models, write transformed text out to a new file, one doc per line
    with codecs.open(filepath_dict['trigram_docs_filepath'], 'w', encoding='utf_8') as f:
        for parsed_doc in nlp.pipe(line_doc(filepath_dict['doc_txt_filepath']),
                                   batch_size=nlp_batch_size,
                                   n_threads=nlp_n_threads):

            # removing punctuation and whitespace
            unigram_doc = [token.lemma_ for token in parsed_doc
                           if not punct_space_more(token)]

            # apply the first-order and second-order phrase models
            bigram_doc = bigram_model[unigram_doc]
            trigram_doc = trigram_model[bigram_doc]

            # remove any remaining stopwords
            trigram_doc = [term for term in trigram_doc
                           if term not in nlp.Defaults.stop_words]

            #extend the stop workds 
            stop_words_extend=['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come']
            trigram_doc = [term for term in trigram_doc
                           if term not in stop_words_extended] 


            # write the transformed doc as a line in the new file
            trigram_doc = ' '.join(trigram_doc)
            f.write(trigram_doc + '\n')
    if verbose:
        logger.info('Done.')

    # put the text back in the dataframe
    trigram_docs = LineSentence(filepath_dict['trigram_docs_filepath'])

    if len([doc for doc in trigram_docs]) == df_phrased.shape[0]:
        for i, doc in enumerate(trigram_docs):
            df_phrased.iloc[i, df_phrased.columns.get_loc(col)] = ' '.join(doc)
    else:
        raise ValueError('Different number of processed and original documents')

    # save dataframe
    if verbose:
        logger.info('Saving NLP processed data: {}'.format(filepath_dict['filepath_out']))
    df_phrased.to_csv(filepath_dict['filepath_out'])

    return df_phrased

def load_phrased_data_pipeline(to_load: str, verbose=True, overwrite_interim=True, prefix=None, training=True, col='resp_whytfa'):
    if prefix is None:
        prefix = ''
    filename_phrased = f'{prefix}{col}_phrased.csv'
    filepath_dict = {}
    filepath_dict['filepath_in'] = None
    if training is True:
        filepath_dict['filepath_out'] = data_dir_training_processed / filename_phrased
    else:
        filepath_dict['filepath_out'] = data_dir_inference_processed / filename_phrased

    df_phrased=pd.read_csv(filepath_dict['filepath_out'])
    return df_phrased

if __name__ == '__main__':
    raw_to_phrased_data_pipeline(to_load, verbose=True, overwrite_interim=True, prefix=None)

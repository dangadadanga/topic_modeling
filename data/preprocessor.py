
from pathlib import Path
import os
import pandas as pd
import numpy as np
import sys
import pickle
# add this module to runtime path
sys.path.append(Path.cwd().parent.as_posix())
import ansaro_utils.data.data_load as au_dl
import ansaro_utils.data.text_utils as au_tu

import logging
logger = logging.getLogger(__name__)

LOCAL_DATA = Path(os.environ['LOCAL_DATA'])
data_dir_local = LOCAL_DATA / 'topic_modeling'
data_dir = Path(data_dir_local)
data_dir_training_processed = data_dir / 'training'/'processed'
if not data_dir_training_processed.exists():
    data_dir_training_processed.mkdir(parents=True, exist_ok=True)
data_dir_inference_processed = data_dir / 'inference'/'processed'
if not data_dir_inference_processed.exists():
    data_dir_inference_processed.mkdir(parents=True, exist_ok=True)



def process_data_whytfa(filepath=None, verbose=True):
    if filepath is None:
        logger.error("No csv file provided")
        exit();
    df = pd.read_csv(filepath)
    df = df.dropna(how='all', subset=[c for c in df if c not in ['TFA Master UID', 'App Year']])
    df = df.drop_duplicates()

    # cols_dict = au_dl.init_cols_dict(df)
    # au_dl.print_cols_dict(cols_dict)

    cols_dict = {
        'TFA Master UID': {
            'class': 'id',
            'col_rename': 'tfa_master_uid',
            'drop': False,
            'dtype': 'int',
            'group': set(),
        },
        'App Year': {
            'class': 'numeric',
            'col_rename': 'app_year',
            'drop': False,
            'dtype': 'int',
            'group': set(),
        },
        'WhyTfa': {
            'class': 'freetext',
            'col_rename': 'resp_whytfa',
            'drop': False,
            'dtype': 'str',
            'group': set(),
            },
        }

    df, cols_dict, cols_list_dict = au_dl.process_df_initial(df, cols_dict, verbose=verbose)
    df = df.drop_duplicates()
    cols_dict, cols_list_dict = au_dl.get_cols_lists(cols_dict)

    # change to int
    df['app_year'] = df['app_year'].astype(str).str.replace(',', '').str.split('.').str[0].astype(int)
    df['tfa_master_uid'] = df['tfa_master_uid'].astype(int)

    # if df.duplicated(subset='tfa_master_uid').sum() > 0:
    #     # WARNING: Some people have multiple applications
    #     warnings.warn('Some people have multiple applications', UserWarning)
    #     warnings.warn('Keeping only one record', UserWarning)
    #     df = df.drop_duplicates(
    #         subset='tfa_master_uid', keep='first')

    response_cols = ['resp_whytfa']
    for col in response_cols:
        # remove applicants who didn't answer why tfa
        if verbose:
            logger.info('Removing {} rows due to not answering Why TFA'.format(
                df[col].isnull().sum()))
        df = df[df[col].notnull()]
        vals_to_remove = ['', 'none', 'na']
        # remove responses that are essentially empty
        if verbose:
            logger.info('Removing invalid responses')
        # remove single-character responses
        df.loc[(df[col].str.len() == 1).fillna(False), col] = np.nan
        # remove records that are in vals_to_remove
        df.loc[df[col].str.replace(r'[^a-zA-Z0-9]+', '').str.lower().isin(vals_to_remove), col] = np.nan
        if verbose:
            logger.info('Processing whytfa response: {}...'.format(col))
        # clean text
        df[col] = df[col].apply(au_tu.fix_text)

    # completely remove people with multiple applications in a single application year
    count_of_apps = df.groupby(by=['tfa_master_uid', 'app_year']).size()
    multiapps = count_of_apps.loc[count_of_apps > 1].reset_index()['tfa_master_uid'].tolist()
    if len(multiapps) > 0:
        logger.info('Removing {} rows due to multiple applications in one year'.format(len(multiapps)))
        df = df.loc[~df['tfa_master_uid'].isin(multiapps)]

    cols_dict, cols_list_dict = au_dl.update_cols_lists(
        cols_dict,
        cols_to_drop='tfa_master_uid',
        remove_dropped=True)

    return df, cols_dict


def process_data_save(filepath=None, as_text=None, as_pickle=None, training=True, verbose=True):
    '''Wrapper to process the data and save to disk. as_text and as_pickle (booleans or None) refer to how to save the data.
    '''
    if as_text is None:
        as_text = True
    if as_pickle is None:
        as_pickle = False
    if not as_text and not as_pickle:
        raise ValueError('Need to set either as_text or as_pickle')

    if verbose:
        logger.info('=' * 30)
        logger.info('Processing data...')
    df, cols_dict = process_data_whytfa(filepath=filepath, verbose=verbose)
    if verbose:
        logger.info('=' * 30)
        logger.info('Processing data: Done')
    if verbose:
        logger.info('=' * 30)
        logger.info('Saving data locally...')
    save_processed_data(df, cols_dict, as_text=as_text, as_pickle=as_pickle, training=training)
    if verbose:
        logger.info('Done')


def save_processed_data(df, cols_dict, as_text=None, as_pickle=None, training=True):
    '''TODO: cols_dict field 'group' is a set, but this isn't JSON serializable. Fix this. Maybe a class with its own save/load methods. For now, always saved as pickle.
    '''
    if as_text is None:
        as_text = True
    if as_pickle is None:
        as_pickle = False
    if not as_text and not as_pickle:
        raise ValueError('Need to set either as_text or as_pickle')
    if training:
        # cols_dict as pickle
        outfile_cols_dict_p = data_dir_training_processed / 'processed_cols_dict.pickle'
        # dataframe as CSV
        outfile_df_csv = data_dir_training_processed / 'processed_dataframe.csv'
        # dataframe as pickle
        outfile_df_p = data_dir_training_processed / 'processed_dataframe.pickle'
    else:
        # cols_dict as pickle
        outfile_cols_dict_p = data_dir_inference_processed / 'processed_cols_dict.pickle'
        # dataframe as CSV
        outfile_df_csv = data_dir_inference_processed / 'processed_dataframe.csv'
        # dataframe as pickle
        outfile_df_p = data_dir_inference_processed / 'processed_dataframe.pickle'
    # write dict pickle
    with open(outfile_cols_dict_p, 'wb') as fp:
        pickle.dump(cols_dict, fp)
    if as_text:
        # write df csv
        df.to_csv(outfile_df_csv)
        # # write dict json
        # with open(outfile_cols_dict_json, 'w') as fp:
        #     json.dump(cols_dict, fp, sort_keys=True, indent=4)
    if as_pickle:
        # write df pickle
        with open(outfile_df_p, 'wb') as fp:
            pickle.dump(df, fp)


def load_processed_data(to_load=None, verbose=True, header=0, index_col=0, training=True):
    if to_load is None:
        to_load = 'text'

    if verbose:
        logger.info('Loading from local directory...')

    if training:
        # cols_dict as pickle
        outfile_cols_dict_p = data_dir_training_processed / 'processed_cols_dict.pickle'
        # dataframe as CSV
        outfile_df_csv = data_dir_training_processed / 'processed_dataframe.csv'
        # dataframe as pickle
        outfile_df_p = data_dir_training_processed / 'processed_dataframe.pickle'
    else:
        # cols_dict as pickle
        outfile_cols_dict_p = data_dir_inference_processed / 'processed_cols_dict.pickle'
        # dataframe as CSV
        outfile_df_csv = data_dir_inference_processed / 'processed_dataframe.csv'
        # dataframe as pickle
        outfile_df_p = data_dir_inference_processed / 'processed_dataframe.pickle'

    # read dict pickle
    with open(outfile_cols_dict_p, 'rb') as fp:
        cols_dict = pickle.load(fp)
    cols_dict, cols_list_dict = au_dl.get_cols_lists(cols_dict)
    if to_load == 'text':
        # read df csv
        df = pd.read_csv(outfile_df_csv, header=header, index_col=index_col)
        # set datetime column type
        for col in cols_list_dict['date']:
            df[col] = pd.to_datetime(df[col])
    elif to_load == 'pickle':
        # read df pickle
        with open(outfile_df_p, 'rb') as fp:
            df = pickle.load(fp)

    if verbose:
        logger.info('Done')
    return df, cols_dict


def response_len(df, response_cols=None, verbose=True):
    '''feature engineer: compute response lengths'''
    # try to drop just in case
    df = df.drop(df.filter(regex=r'^resp_(.*?)_len$').columns, axis=1)
    df = df.drop(df.filter(regex=r'^resp_(.*?)_unique_words$').columns, axis=1)

    if response_cols is None:
        response_cols = df.filter(regex='^resp_').describe().loc['count'].sort_values(ascending=False).index.tolist()

    if verbose:
        logger.info('Calculating response lengths... ')

    # calculate response length
    for col in response_cols:
        df[f'{col}_len'] = df[col].apply(lambda x: au_tu.count_words(x))

    # count number of unique words
    for col in response_cols:
        df[f'{col}_unique_words'] = df[col].apply(lambda x: au_tu.count_words(x, unique=True))
    if verbose:
        logger.info('Done.')
    return df


def remove_shortest_essays(df, thresh=0.0025, response_cols=None, verbose=True):
    '''Set short response essays in the given bottom quantile to null (np.nan)
    '''
    if response_cols is None:
        response_cols = df.filter(regex=r'^resp_(.*?)((?<!_len)+(?<!_unique_words))$').columns.tolist()

    # remove shortest responses by word length
    if verbose:
        if thresh < 1:
            logger.info(f'Removing shortest {thresh:.3%} responses')
        else:
            logger.info(f'Removing responses with fewer than {thresh} words')
    for col in response_cols:
        len_col = f'{col}_len'
        if thresh < 1:
            # quantile
            thresh_val = int(np.ceil(df[len_col].quantile(q=thresh)))
            if verbose:
                logger.info(f'\tFewer than {thresh_val} words')
        else:
            thresh_val = thresh
        if verbose:
            logger.info(f'\t{col}: {df.loc[df[len_col] < thresh_val].shape[0]} responses')
        df.loc[df[len_col] < thresh_val, col] = np.nan
    return df

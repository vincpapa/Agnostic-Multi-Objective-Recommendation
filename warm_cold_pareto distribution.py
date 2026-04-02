import pandas as pd
import numpy as np

datasets = ['amazon_book']

for dataset in datasets:
    train = pd.read_csv(f'data/{dataset}/train.tsv', sep='\t', header=None)
    train_80 = round(len(train) * 0.8)
    user_count = train.groupby(0).size().reset_index(name='counts').sort_values(by='counts',
                                                                                ascending=True).reset_index(drop=True)
    item_count = train.groupby(1).size().reset_index(name='counts').sort_values(by='counts',
                                                                                ascending=True).reset_index(drop=True)

    user_count['group'] = pd.Series(np.zeros(len(user_count), dtype='int'))
    item_count['group'] = pd.Series(np.zeros(len(item_count), dtype='int'))

    # 0: cold user/item
    # 1: warm user/item
    
    cumulative_sum = 0
    for ind in user_count.index:
        cumulative_sum += user_count['counts'][ind]
        if cumulative_sum >= train_80:
            user_count.at[ind, 'group'] = 1

    cumulative_sum = 0
    for ind in item_count.index:
        cumulative_sum += item_count['counts'][ind]
        if cumulative_sum >= train_80:
            item_count.at[ind, 'group'] = 1

    user_count[[0, 'group']].to_csv(f'data/{dataset}/user_groups.tsv', sep='\t', header=None, index=None)
    item_count[[1, 'group']].to_csv(f'data/{dataset}/item_groups.tsv', sep='\t', header=None, index=None)

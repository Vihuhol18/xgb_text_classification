import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt



def create_df(dir, est):
    listdir = os.listdir(dir)
    list_reviews = []
    for i_file in listdir:
        if i_file[-3:] == 'txt':
            with open(f'{dir}/{i_file}', 'r') as f:
                text = f.read()
                id = i_file[:-4].split(sep='_')
                rating = id[-1]
                id = id[0]
                l_text = len(text.split())
                list_reviews.append([text, id, l_text, rating, est])

    df_reviews = pd.DataFrame(list_reviews, columns=['text', 'id', 'l_text', 'rating', 'estimate'])

    df_reviews = df_reviews.astype({'id': 'int64', 'l_text': 'int64',
                                    'rating': 'int32', 'estimate': 'int32'})

    # print(df_reviews.dtypes)

    df_reviews = df_reviews.sort_values(by='id')


    return df_reviews




df_test_neg = create_df('aclImdb/test/neg', 0)
df_test_pos = create_df('aclImdb/test/pos', 1)

df_test = pd.concat([df_test_neg, df_test_pos], ignore_index=True)

df_train_neg = create_df('aclImdb/train/neg', 0)
df_train_pos = create_df('aclImdb/train/pos', 1)

df_train = pd.concat([df_train_neg, df_train_pos], ignore_index=True)


fig, axs = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Test data analysis')
sns.histplot(data=df_test, x="rating", ax=axs[0])
box = axs[0].get_position()
axs[0].set_position([box.x0-0.03, box.y0, box.width, box.height])
sns.histplot(data=df_test, x="estimate", ax=axs[1])
box = axs[2].get_position()
axs[2].set_position([box.x0, box.y0, box.width, box.height])
sns.histplot(data=df_test, x="l_text", hue="estimate", ax=axs[2])
box = axs[2].get_position()
axs[2].set_position([box.x0+0.03, box.y0, box.width, box.height])
plt.savefig('little_analysis_test.png', dpi=300)


fig, axs = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Train data analysis')
sns.histplot(data=df_train, x="rating", ax=axs[0])
box = axs[0].get_position()
axs[0].set_position([box.x0-0.03, box.y0, box.width, box.height])
sns.histplot(data=df_train, x="estimate", ax=axs[1])
box = axs[2].get_position()
axs[2].set_position([box.x0, box.y0, box.width, box.height])
sns.histplot(data=df_train, x="l_text", hue="estimate", ax=axs[2])
box = axs[2].get_position()
axs[2].set_position([box.x0+0.03, box.y0, box.width, box.height])
plt.savefig('little_analysis_train.png', dpi=300)



# df_test.to_csv('df_test')
# df_train.to_csv('df_train')



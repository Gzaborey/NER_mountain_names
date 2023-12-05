import pandas as pd
from simpletransformers.ner import NERModel, NERArgs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


if __name__ == '__main__':

    # DATA PREPARATION
    ''' The purpose of this section is to prepare the data for training.
        Firstly, the data is loaded and renamed to match the model requirements.
        Secondly, the data is split into train and text sets.
        Due to the lack of training data, there will be no validation set.'''

    df_path = 'https://raw.githubusercontent.com/Gzaborey/test_task/main/NER_mountain_names/mountain_names.csv'
    df = pd.read_csv(df_path, index_col=0, encoding='UTF-8')
    df = df.rename(columns={'tokens': 'words', 'ner_tags': 'labels'})

    X = df.loc[:, ['words', 'sentence_id']]
    y = df.labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    train_df = pd.DataFrame({'sentence_id': X_train.sentence_id,
                             'words': X_train.words,
                             'labels': y_train})
    test_df = pd.DataFrame({'sentence_id': X_test.sentence_id,
                            'words': X_test.words,
                            'labels': y_test})

    # MODEL TRAINING

    # extracting the custom labels to feed them to the model later
    labels = df["labels"].unique().tolist()

    # initialising the model's arguments
    args = NERArgs()
    args.num_train_epochs = 10
    args.learning_rate = 1e-4
    args.overwrite_output_dir = True
    args.save_steps = -1  # do not save results of intermediate training steps
    args.save_model_every_epoch = False  # # do not save results of intermediate training epochs
    args.train_batch_size = 16
    args.eval_batch_size = 16

    # loading pre-trained BERT model
    model = NERModel('bert', 'bert-base-cased', labels=labels,
                     args=args, use_cuda=True)  # CUDA needs to be available
    model.train_model(train_df, eval_data=test_df, acc=accuracy_score)

    # evaluating the model performance
    result, model_outputs, preds_list = model.eval_model(test_df)

    print(result)

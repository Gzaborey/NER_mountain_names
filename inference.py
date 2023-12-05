from simpletransformers.ner import NERModel


if __name__ == '__main__':

    text = 'Is Hoverla the highest peak in Ukraine?'  # enter the text to analyze
    model_path = 'outputs'  # enter the path to the saved model

    model = NERModel('bert', 'outputs')
    predictions = model.predict(['Is Hoverla the highest peak in Ukraine?'])

    print(predictions)

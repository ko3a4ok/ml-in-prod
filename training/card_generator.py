import model_card_toolkit as mct

def generate_model_card_html():
  model_card_output_path = 'card_output/'
  toolkit = mct.ModelCardToolkit(model_card_output_path)
  model_card = toolkit.scaffold_assets()
  model_card.model_details.name = 'My L5 Model'
  model_card.model_details.overview = 'Model is to rate the complexity of literary passages for grades 3-12 classroom use'
  model_card.model_details.documentation = '''
  Model is created using the Keras framework as a standard sequence model.
  Input for the model is a text, and the output is a 1 float number - the prediction score.
  '''
  model_card.model_details.licenses.append(mct.License(identifier='0BSD'))
  model_card.model_details.owners.append(
      mct.Owner(name='Roma', contact="Zhytomyr"))
  model_card.model_parameters.model_architecture = 'Sequence model with text input and 1 float output'
  model_card.model_parameters.input_format = 'Text'
  model_card.model_parameters.output_format = 'Float between 0 and 1, represents the score'
  model_card.model_parameters.data.append(
      mct.Dataset(name='CommonLit Readability Prize',
                  link='https://www.kaggle.com/competitions/commonlitreadabilityprize/data',
                  description='''
  The dataset is used in the competition, where the goal is a predicting the 
  reading ease of excerpts from literature. We've provided excerpts from several
  time periods and a wide range of reading ease scores. Note that the test set 
  includes a slightly larger proportion of modern texts (the type of texts we want
  to generalize to) than the training set.
  '''))
  model_card.quantitative_analysis.performance_metrics.append(
      mct.PerformanceMetric(type='Accuracy', value='0.7', slice='Test dataset'))
  toolkit.update_model_card(model_card)
  html = toolkit.export_format()
  print(html)


if __name__ == '__main__':
    generate_model_card_html()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mlxtend import plotting as mlx_plotting, evaluate as mlx_evaluate
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3

''' Links de apoio:
    Artigo = https://www.hindawi.com/journals/jhe/2020/8860011/
    Keras InceptionV3 = https://keras.io/api/applications/inceptionv3/
    Keras Transfer Learning = https://keras.io/guides/transfer_learning/
    Pooling Layers = https://www.machinecurve.com/index.php/2020/01/30/what-are-max-pooling-average-pooling-global-max-pooling-and-global-average-pooling/
    Plota uma representação gráfica do modelo = keras.utils.plot_model(base_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True) #Cria uma representação gráfica do modelo
'''

converted_dataset_root_path = '/media/rogerio/49a45a42-45ee-49c3-9085-9defff31abe8/TCC-Dataset/code/dataset/converted/'
training_dataset_path = 'Mass-Training-Full-Mammogram-Images-(DICOM)'
test_dataset_path = 'Mass-Test-Mammogram-Images-(DICOM)'
model_test_dataset_true_classification_path = '/media/rogerio/49a45a42-45ee-49c3-9085-9defff31abe8/TCC-Dataset/code/model_test_dataset_true_classification'
class_names =   ['BENIGN','BENIGN_WITHOUT_CALLBACK','MALIGNANT']
image_size = (300, 300) #(299, 299) #(256, 512) #(512, 1024)

# ------------------------------------------------------------ Model Processing ----------------------------------------------------------- 

#1 Step = Ler o dataset em RGB
print('========================================== STARTING  ==========================================')
print('Reading trainig dataset ...............')
images_train_dataset = preprocessed_data = keras.preprocessing.image_dataset_from_directory(
    '{0}/{1}'.format(converted_dataset_root_path, training_dataset_path),
    labels="inferred",
    label_mode="categorical",
    class_names=class_names,
    color_mode="rgb",
    batch_size=32,
    image_size=image_size,
    shuffle=True,
    seed=1337,
    validation_split=0.2,
    subset='training',
    interpolation="bilinear",
    follow_links=False,
)
print('Reading validation dataset ...............')
images_validation_dataset = preprocessed_data = keras.preprocessing.image_dataset_from_directory(
    '{0}/{1}'.format(converted_dataset_root_path, training_dataset_path),
    labels="inferred",
    label_mode="categorical",
    class_names=class_names,
    color_mode="rgb",
    batch_size=32,
    image_size=image_size,
    shuffle=True,
    seed=1337,
    validation_split=0.2,
    subset='validation',
    interpolation="bilinear",
    follow_links=False,
)
print('Reading test dataset ...............')
images_test_dataset = preprocessed_data = keras.preprocessing.image_dataset_from_directory(
    '{0}/{1}'.format(converted_dataset_root_path, test_dataset_path),
    labels="inferred",
    label_mode="categorical",
    class_names=class_names,
    color_mode="rgb",
    batch_size=32,
    image_size=image_size,
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
)
print('Reading test dataset true classification response ...............')
model_test_dataset_true_classification = pickle.load(open(model_test_dataset_true_classification_path,"rb"))
#2 Step = Carregar o modelo InceptionV3 pré treinado com ImageNet e remove a camada densa (fully-connected layer) softmax default da rede
base_model = InceptionV3(include_top=False, weights='imagenet',input_shape=(300, 300, 3), classes=3)
#3 Step = Adiciona uma camada de Max Pooling Global e uma camada densa para padronizar a saída multiclasse
outputs = keras.layers.GlobalAveragePooling2D()(base_model.layers[-1].output)
outputs = keras.layers.Dense(3, activation='softmax')(outputs)
model = keras.Model(inputs=base_model.input, outputs=outputs)
#5 Step = Loga no console o resumo do modelo final
print(model.summary())
#6 Step = Utiliza a função de erro entropia cruzada e define a taxa de aprendizado para um valor abaixo do que foi utilizado no treinamento anterior para manter o conhecimento previamnte adquirido pelo InceptionV3
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(1e-5), metrics=['accuracy'])
#7 Step = Realiza o treinamento em 30 epocas e lê 128 entradas até realizar o backpropagation
classifier = model.fit(images_train_dataset, validation_data=images_validation_dataset, batch_size=128, epochs=10) #128, 20
#8 Step = Salva o modelo treinado no disco
model.save('model_fitted.h5')
#9 Step = Realiza a predição no dataset de teste
predicted_result = model.predict(images_test_dataset)
#10 Step = Padroniza a saída em valores inteiros para utilizarmos calcularmos as métricas ([1,0,0](Benigno) = 0 | [0,1,0](Benigno sem retorno) = 1 | [0,0,1] (Maligno) = 2)
# Converte a resposta da predição de um array bidimensional de numeros com ponto flutuante (float32) para um array unidimensional de números inteiros (int)
print('Normalizing prediction output to unidimensional vector of integers ...............')
predicted_result = np.argmax(predicted_result, axis=1)
print('Model predict output............... : {0}'.format(predicted_result))

print('Salvando resultado do treinamento ...............')
pickle_out = open('model_training_result', "wb")
pickle.dump(classifier.history, pickle_out)
pickle_out.close()
print('Salvando resultado da predição ...............')
pickle_out = open('model_predicted_result', "wb")
pickle.dump(predicted_result, pickle_out)
pickle_out.close()

# ------------------------------------------------------------ Metrics Evaluation ----------------------------------------------------------- 

# ------------------- Printing Accuracy Image ------------------- 
print('Accuracy...............: ', classifier.history['accuracy'])
plt.figure()
plt.title('Acurácia x Epocas - Treinamento')
plt.xlabel('Epocas')
plt.ylabel('Acurácia')
plt.plot(classifier.history['accuracy'])
plt.plot(classifier.history['val_accuracy'])
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('model_multiclass_accuracy_classifier.png')

# ------------------- Printing Error Image ------------------- 
print('Accuracy...............: ', classifier.history['accuracy'])
plt.figure()
plt.title('Erro x Epocas - Treinamento')
plt.xlabel('Epocas')
plt.ylabel('Erro')
plt.plot(classifier.history['loss'])
plt.plot(classifier.history['val_loss'])
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('model_multiclass_error_classifier.png')

# ------------------- Printing Multiclass Confusion Matrix Report -------------------
# Documentação: http://rasbt.github.io/mlxtend/user_guide/evaluate/confusion_matrix/#example-2-multi-class-classification
print('Metrics - Multiclass Confusion Matrix Report ...............:')
cm = confusion_matrix(model_test_dataset_true_classification, predicted_result)
fig, ax = mlx_plotting.plot_confusion_matrix(conf_mat=cm)
plt.savefig('model_multiclass_confusion_matrix.png')

# ------------------- Printing Classification Report ------------------- 
#   Documentação: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
print('Metrics - Classification Report ...............:')
print(classification_report(model_test_dataset_true_classification, predicted_result, target_names=class_names))

# ------------------- Binarizing Prediction for metrics evaluation ------------------- 
print('Binarizing multiclass prediction response for metrics evaluation ...............:')
#   Classe 1 vs Classes 2 e 3
#   Classe 2 vs Classes 1 e 3
#   Classe 3 vs Classes 1 e 2

binary_prediction = {
    'classification': [],
    'prediction': [],
    'confusion_matrix': []
}

for index in range(len(class_names)):
    #Converte os valores do vetor que são diferentes da classe atual para 0 e o valor que da classe para 1
    binary_prediction['classification'].append(np.where(model_test_dataset_true_classification != index, 0, 1))
    binary_prediction['prediction'].append(np.where(predicted_result != index, 0, 1))

    # ------------------- Printing Confusion Matrix ------------------ 
    #   Documentação: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    binary_prediction['confusion_matrix'].append(confusion_matrix(binary_prediction['classification'][index], binary_prediction['prediction'][index]))
    print('Confusion Matrix for {} class...............: {}'.format(class_names[index], binary_prediction['confusion_matrix'][index]))
    false_labels = list(filter(lambda x: x != class_names[index], class_names))
    ConfusionMatrixDisplay(
        confusion_matrix=binary_prediction['confusion_matrix'][index],
        display_labels=[class_names[index], '{}/{}'.format(false_labels[0], false_labels[1])]
    ).plot()
    plt.savefig('model_multiclass_confusion_matrix_{}.png'.format(class_names[index]))
    
    # ------------------- Printing ROC Curve ------------------- 
    # Documentation: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    fpr = dict() #falsos positivos
    tpr = dict() #verdadeiros positivos
    roc_auc = dict()
    fpr[index], tpr[index], thresholds = roc_curve(binary_prediction['classification'][index], binary_prediction['prediction'][index])
    print('ROC Curve for {} class...............: True Positive = {}, False Positive = {}'.format(class_names[index], tpr[index], fpr[index]))
    roc_auc[index] = auc(fpr[index], tpr[index])
    print('Area Under the Curve for {} class...............:'.format(class_names[index]), roc_auc[index])

    plt.figure()
    plt.plot(fpr[index], tpr[index], label='ROC curve (area = %0.2f)' % roc_auc[index])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{} - ROC Curve'.format(class_names[index]))
    plt.legend(loc="lower right")
    plt.savefig('model_multiclass_roc_curve_{}.png'.format(class_names[index]))
from apps.ml.income_classifier.textGenerator import TextGenerator
from apps.ml.income_classifier.classifier2 import Classifier2
from apps.ml.income_classifier.naiveBayes import NaiveBayesClassifier
import pickle
import os
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from keras.models import model_from_json
from scipy.sparse import csr_matrix
import numpy as np
from tensorflow.keras import backend as K
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate

class MLModel:

    def __init__(self): 
        CURRENT_DIR = os.path.dirname(__file__)
        TEMPLATE_DIRS = (
            os.path.join(CURRENT_DIR, '../../../machineLearning/')
        )
        self.model_classifier1 = pickle.load(open(TEMPLATE_DIRS + "naive_bayes.pkl", "rb"))
        self.vectorizer_from_train_data_classifier1 = pickle.load(open(TEMPLATE_DIRS + "count_vectorizer.pkl", "rb"))

        with open(TEMPLATE_DIRS + "model.json", 'rb') as json_file:
            self.loaded_model_json = json_file.read()
        json_file.close()
        self.in_tokenizer = pickle.load(open(TEMPLATE_DIRS + "in_tokenizer.pkl", "rb"))
        self.tr_tokenizer = pickle.load(open(TEMPLATE_DIRS + "tr_tokenizer.pkl", "rb"))
        self.loaded_model = model_from_json(self.loaded_model_json)
        self.loaded_model.load_weights(TEMPLATE_DIRS + "model.h5")

        self.model_classifier2 = pickle.load(open(TEMPLATE_DIRS + "naive_bayes_2.pkl", "rb"))
        self.vectorizer_from_train_data_classifier2 = pickle.load(open(TEMPLATE_DIRS + "count_vectorizer_2.pkl", "rb"))

    # the method applies pre-processing
    def preprocessing_classifier1(self, input_data):
        words = input_data["paragraph"]
        data = self.vectorizer_from_train_data_classifier1.transform([words])
        # JSON to array
        return csr_matrix.toarray(data)

    # the method that calls ML for computing predictions on prepared data
    def predict_classifier1(self, input_data):
        return self.model_classifier1.predict_proba(input_data)

    # the method that applies post-processing on prediction values
    def postprocessing_classifier1(self, input_data):
        label = "false"
        if input_data[1] > 0.5:
            label = "true"
        return {"proba": label, "label": input_data, "status": "OK"}

    # the method that combines: preprocessing, predict and postprocessing and returns JSON object with the response
    def compute_prediction_classifier1(self, input_data):
        try:
            input = self.preprocessing_classifier1(input_data)
            prediction = self.predict_classifier1(input)[0]
            prediction = self.postprocessing_classifier1(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}
        return prediction

    # Method to create clean text
    def clear_text_generator(self, text):
        txt = text["paragraph"]
        cleanText = pad_sequences(
            self.in_tokenizer.texts_to_sequences([self.preprocess_text_text_generator(txt)]),
            padding='post',
            maxlen=312
        )
        return cleanText

    # predict method
    def predict_text_generator(self, input_data):
        prediction = self.decode_sequence_text_generator((self.clear_text_generator(input_data)).reshape(1,312))
        return prediction

    # the method that applies post-processing on prediction values
    def postprocessing_text_generator(self, input_data):
        return {"summary": input_data, "status": "OK"}

    # the method that combines: predict and postprocessing and returns JSON object with the response
    def compute_prediction_text_generator(self, input_data):
        try:
            prediction = self.predict_text_generator(input_data)
            prediction = self.postprocessing_text_generator(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}
        return prediction

    # the method that generates the summary
    def decode_sequence_text_generator(self, input_seq):
        K.clear_session() 
        latent_dim=500
        max_tr_len=50
        # Construct encoder model from the output of 6 layer i.e.last LSTM layer
        en_outputs,state_h_enc,state_c_enc = self.loaded_model.layers[6].output
        en_states=[state_h_enc,state_c_enc]
        # Add input and state from the layer.
        en_model = Model(self.loaded_model.input[0],[en_outputs]+en_states)
        # Decoder inference
    
        dec_state_input_h = Input(shape=(latent_dim,))
        dec_state_input_c = Input(shape=(latent_dim,))
        dec_hidden_state_input = Input(shape=(312,latent_dim))
        
        # Get the embeddings and input layer from the model
        dec_inputs = self.loaded_model.input[1]
        dec_emb_layer = self.loaded_model.layers[5]
        dec_lstm = self.loaded_model.layers[7]
        dec_embedding= dec_emb_layer(dec_inputs)
        
        # Add input and initialize LSTM layer with encoder LSTM states.
        dec_outputs2, state_h2, state_c2 = dec_lstm(dec_embedding, initial_state=[dec_state_input_h,dec_state_input_c])

        # Attention layer
        attention = self.loaded_model.layers[8]
        attn_out2 = attention([dec_outputs2,dec_hidden_state_input])
        
        merge2 = Concatenate(axis=-1)([dec_outputs2, attn_out2])

        # Dense layer
        dec_dense = self.loaded_model.layers[10]
        dec_outputs2 = dec_dense(merge2)
        
        # Finally define the Model Class
        dec_model = Model(
        [dec_inputs] + [dec_hidden_state_input,dec_state_input_h,dec_state_input_c],
        [dec_outputs2] + [state_h2, state_c2])

        reverse_target_word_index = self.tr_tokenizer.index_word
        reverse_source_word_index = self.in_tokenizer.index_word
        target_word_index = self.tr_tokenizer.word_index
        reverse_target_word_index[0]=' '

        en_out, en_h, en_c= en_model.predict(input_seq)
 
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1))
        
        # Target sequence with initial word as 'start'
        target_seq[0, 0] = target_word_index['start']
    
        # If the iteration reaches the end of text than it will be stop the iteration
        stop_condition = False
        
        # Append every predicted word in decoded sentence
        decoded_sentence = ""
        while not stop_condition: 
            # Get predicted output, hidden and cell state.
            output_words, dec_h, dec_c= dec_model.predict([target_seq] + [en_out,en_h, en_c])
            
            # Get the index and from the dictionary get the word for that index.
            word_index = np.argmax(output_words[0, -1, :])
            text_word = reverse_target_word_index[word_index]
            decoded_sentence += text_word +" "
            
            # Exit condition: either hit max length or find a stop word or last word.
            if text_word == "end" or len(decoded_sentence) > max_tr_len:
                stop_condition = True
                
            # Update target sequence to the current word index.
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = word_index
            en_h, en_c = dec_h, dec_c
        # Return the decoded sentence
        return decoded_sentence

    # Method to do some preprocessing
    def preprocess_text_text_generator(self, sen):
        # Removing html tags
        # sentence = remove_tags(sen)
        # Remove punctuations and numbers
        # sentence = re.sub('[^a-zA-Z]', ' ', sentence)
        # Single character removal
        # sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
        # Removing multiple spaces
        # sentence = re.sub(r'\s+', ' ', sentence)
        return sen
    
    # the method applies pre-processing
    def preprocessing_classifier2(self, input_data):
        # words = input_data["paragraph"]
        data = self.vectorizer_from_train_data_classifier2.transform([input_data])
        # JSON to array
        return csr_matrix.toarray(data)

    # the method that calls ML for computing predictions on prepared data
    def predict_classifier2(self, input_data):
        return self.model_classifier2.predict(input_data)

    # the method that applies post-processing on prediction values
    def postprocessing_classifier2(self, input_data):
        return {"label": input_data, "status": "OK"}

    # the method that combines: preprocessing, predict and postprocessing and returns JSON object with the response
    def compute_prediction_classifier2(self, input_data):
        try:
            input = self.preprocessing_classifier2(input_data)
            prediction = self.predict_classifier2(input)[0]
            prediction = self.postprocessing_classifier2(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}
        return prediction

    def mlmodel(self, input_data):
        try:
            prediction = self.compute_prediction_classifier1(input_data)
            if (prediction['proba'] =='true'):
                text = self.compute_prediction_text_generator(input_data)
                summary = text['summary']
                classification = self.compute_prediction_classifier2(summary)
                return {"paragraph":input_data['paragraph'],"summary": text['summary'], "label": classification['label'], "status": "OK"}
            else:
                return{"input doesn't contain IE"}
        except Exception as e:
            return {"status": "Error", "message": str(e)}

    # def mlmodeltext(self,text):
    #     try:
    #         # Import the stop words of the English language 
    #         stop = stopwords.words('english')
    #         # Normalization
    #         text = text.lower()
    #         # Split text into paragraphs 
    #         data= text.split("\n")
    #         print("Number of paragraphs is : ", len(data))
    #         # delete empty lines from the document 
    #         for index, p in enumerate(data): 
    #             if data[index] == "":
    #                 del data[index]
                
    #         print("Number of paragraphs is : ", len(data))

    #         # start cleaning 
    #         for index, p in enumerate(data): 
    #         # Remove unicode chars
    #             data[index] = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", p)
    #             # Remove stop words 
    #             # data[index] = " ".join([word for word in data[index].split() if word not in (stop)])
    #             # Stemming 
    #             # words = data[index].split()
    #             # stemmer = PorterStemmer()
    #             # for i, word in enumerate(words):
    #             #   words[i] = stemmer.stem(word)
    #             # newPragraphe = " ".join(words)
    #             # data[index] = " ".join(words)
    #         print(data)
    #         return {"data": data, "status": "OK"}
    #     except Exception as e:
    #         return {"status": "Error", "message": str(e)}

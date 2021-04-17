from flask import *  
import gensim
import cv2
import os
model=gensim.models.Word2Vec.load("final_text_to_text.h5")


app = Flask(__name__)  


@app.route('/')  
def upload():  
    return render_template("nlp_text.html")  
 
    
@app.route('/upload_image', methods = ['POST'])  
def upload_image():   
    in_features = [x for x in request.form.values()]
    prediction=model.wv.most_similar(positive=in_features, topn=5)
    new_lst=[]
    for i in prediction:
        new_lst.append(i[0])  
        
    return render_template("nlp_text.html", display = new_lst)  

if __name__ == '__main__':  
    app.run(debug = True) 

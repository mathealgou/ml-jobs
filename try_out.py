import pickle
import os
from sklearn.preprocessing import MultiLabelBinarizer

def try_out():
    print("try out the model for yourself!")
    
    # load the model from disk
    filename = 'finalized_model.sav'
    classifier = pickle.load(open(filename, 'rb'))
    mlb = pickle.load(open('finalized_transformer.sav', 'rb'))
    
    while True:
        skills = input('Enter the skills required for the job: (separated by "," )')
        skills = skills.split(',')
    
        skills = [skill.strip() for skill in skills]
        skills = [skill.lower() for skill in skills]
        
        test_skills = mlb.transform([skills])
        predicted_job = classifier.predict(test_skills)
        print("predicted job: ", predicted_job)
        
        print("press q to quit or enter to continue")
        if input() == 'q':
            break
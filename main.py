from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
import csv
import os
import pandas as pd
import pickle
import warnings
from fitness import get_fitness
from try_out import try_out

warnings.filterwarnings("ignore") # Necessary to ignore warnings because of sklearn behavior (check https://stackoverflow.com/questions/72060723/i-am-having-a-userwarning-unknown-classes-when-using-multilabelbinarizer)


TEST_DATA_PERCENTAGE = 0.05


def main():
	print("trying to load model and transformer from disk")

	try:
		classifier = pickle.load(open('finalized_model.sav', 'rb'))
		mlb = pickle.load(open('finalized_transformer.sav', 'rb'))
		print("model and transformer loaded from disk")
		try_out()
		return
	except:
		print("error loading model and transformer from disk")
		print("training model from scratch")
		pass
			
	# load scrapped data
	scrapped_data = csv.reader(open(os.path.join(os.path.dirname(__file__), 'scrapped_data.csv'), 'r'))
	
 	# transform into pandas dataframe
	jobs_df = pd.DataFrame(scrapped_data, columns=["Job Name","Company Name","JD","Skills","Date Posted","YOE","Location","Website","Job Function:","Industry:","Specialization:","Qualification:","Hiring Location:","Role:","Vacancies:"])
	jobs_df = jobs_df.dropna()

	# remove unuseful columns
	jobs_df = jobs_df.drop(columns=['Company Name', 'JD', 'Date Posted', 'YOE', 'Location', 'Website', 'Job Function:', 'Industry:', 'Specialization:', 'Qualification:', 'Hiring Location:', 'Role:', 'Vacancies:'])

	train_data = jobs_df.sample(frac=1-TEST_DATA_PERCENTAGE, random_state=200)
	test_data = jobs_df.drop(train_data.index)

	print("training data length", len(train_data))
	print("testing data length", len(test_data))

	# convert skills to a binary matrix
	skills = [skills.split(',') for skills in train_data['Skills']]
 
	mlb = MultiLabelBinarizer()
	skills_binary = mlb.fit_transform(skills)

	# train a multi-label classifier
	classifier = RandomForestClassifier()
	classifier.fit(skills_binary, train_data['Job Name'])
 
	# save the model to disk
	filename = 'finalized_model.sav'
	pickle.dump(classifier, open(filename, 'wb'))
	transformer_filename = 'finalized_transformer.sav'
	pickle.dump(mlb, open(transformer_filename, 'wb'))


	# test the model 
	test_data = test_data[['Skills', 'Job Name']]

	# loop through each row and predict job name
	fitness_scores = []
	for index, row in test_data.iterrows():
		test_skills = mlb.transform([row['Skills'].split(',')])
		predicted_job = classifier.predict(test_skills)
		fitness_score = get_fitness(predicted_job[0], row['Job Name']) * 100
		fitness_scores.append(fitness_score)
	average_fitness_score = sum(fitness_scores) / len(fitness_scores)
	print("average fitness score: ", average_fitness_score)

	# try out the model for yourself
	try_out()


if __name__ == '__main__':
	main()
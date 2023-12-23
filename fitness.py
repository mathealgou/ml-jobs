import re

def clean_string(string):
	string = string.lower()
	string = re.sub(r'[^a-zA-Z ]+', '', string)
	string = string.split(' ')
	string = [word for word in string if len(word) > 0]
	return string

def get_fitness(predicted_job, actual_job):
	# cleanup
	clean_predicted_job = clean_string(predicted_job)
	clean_actual_job = clean_string(actual_job)
 	# Calculate how many words are in common between the predicted job and the actual job
	common_words = set(clean_predicted_job).intersection(clean_actual_job)
 
 	# Calculate the fitness score
	fitness_score = len(common_words) / len(clean_actual_job)
	return fitness_score
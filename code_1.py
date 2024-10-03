import numpy as np
import pandas as pd
import sklearn
import nltk
import pickle

# importing the data sets   

movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')

#DATA PRE-PROCESSING 

#merging the dataframes 
movies=movies.merge(credits,on='title')

#dropiing the useless data values

movies=movies[['movie_id','title','genres','overview','keywords','cast','crew']]
#print(movies.info()) 
# dropping the null values
movies.dropna(inplace=True)

#removing the wierdness in data 
import ast
# the ast module is used to convert the string into a list

def convert(obj): #helper function to convert
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name']) 
    return L

movies['genres']=movies['genres'].apply(convert)
#print(movies['genres'])

movies['keywords']=movies['keywords'].apply(convert)

#print(movies['keywords'])

def convert_cast(obj): #helper function to convert string to list
    L=[]          # and get the top three actors 
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
             L.append(i['name']) 
             counter+=1
        else:
            break
    return L

movies['cast']=movies['cast'].apply(convert_cast)

#print(movies['cast'])

def get_director(obj): #helper function to get the director name
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name']) 
            break
    return L

movies['crew']=movies['crew'].apply(get_director)
#print(movies['crew'])

# removing the spaces from the data 
    

movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])   
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x]) 
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

 # creating a tags collum by merging cast,crew,overview and keywords

movies['tags'] = movies['genres']+movies['keywords']+movies['cast']+movies['crew']
# now creating a new data frame with id,title and tags collumn
new_df = movies[['movie_id','title','tags']]

#convert the dataframe from list to string 
movies['overview']=movies['overview'].apply(lambda x:" ".join(x))
new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))
new_df['tags']=movies['overview']+new_df['tags']


# convert to lower case 

new_df['tags']=new_df['tags'].apply(lambda x:x.lower())
#print(new_df['tags'])

# now the data set is ready and now we will use  
# text vectorization and calculate the similarity score to recommend the movies
# we will use bag of words technique to convert them into vectors 

from sklearn.feature_extraction.text import CountVectorizer

#create the object for this class
 
cv=CountVectorizer(max_features=9000,stop_words='english')
# crearting the vectors using numpy method
vectors=cv.fit_transform(new_df['tags']).toarray()

# apply stemming 
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
def stem(text):
    Y=[]
    for i in text.split():
        ps.stem(i)
    return " ".join(Y)
new_df['tags'].apply(stem)

# now we will calculate the distance of each vector from other vectors
# but we will not calculate the euclidean distance because for high dimensional data it fails
# instead we will calculate the cosine distance or we can say the angle between two vector 

from sklearn.metrics.pairwise import cosine_similarity
similarity=cosine_similarity(vectors)

#sorting the similarity score 
sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# now creating a function to recommend movies 

def recommend(movie):
    #getting the index of the movie 

    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similarity[movie_index]
    movies_list= sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)

pickle.dump(new_df,open('movie_list.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))














  
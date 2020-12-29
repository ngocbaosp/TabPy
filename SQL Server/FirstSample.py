# imports for SQL data part
import pyodbc
from datetime import datetime, timedelta
import pandas as pd
import requests

################################################################################
def updateMovieById(id,image):
    cnxn_str = ("Driver={SQL Server Native Client 11.0};"
                "Server=BAOTRAN;"
                "Database=Python;"
                "Trusted_Connection=yes;")

    cnxn = pyodbc.connect(cnxn_str)  # initialise connection (assume we have already defined cnxn_str)

    query = "UPDATE dbo.tmdb_5000_movies_img SET overview ='"+image+"' WHERE Id = "+str(id)

    cnxn.execute(query)

    cnxn.commit()

    del cnxn  # close the connection

def getImageById(id):
    url = 'https://api.themoviedb.org/3/movie/' + str(id)
    token = '?api_key=a2098addbce9ac018efa26ad55ff207c'
    url = url + token
    jData = requests.get(url).json()
    result = ''

    print(jData)

    #for key, value in j.items():
    #    print(key,value)
    try:
        #print(j['poster_path'])
        result = jData['poster_path']
    except:
        print('Error Error')
    return result

###################################
def getAndUpdateImagePath(movieId):
    imagePath = getImageById(movieId)  # 157336
    print(imagePath)
    try:
        updateMovieById(movieId, imagePath)
    except:
        print('error')


movieId = 310706
getAndUpdateImagePath(movieId)


def test():
    cnxn_str = ("Driver={SQL Server Native Client 11.0};"
                "Server=BAOTRAN;"
                "Database=Python;"
               "Trusted_Connection=yes;")

    cnxn = pyodbc.connect(cnxn_str)  # initialise connection (assume we have already defined cnxn_str)

    # build up our query string
    #query = ("SELECT * FROM Employee ")
    #query = ("SELECT id,original_title FROM dbo.tmdb_5000_movies_img ")

    query = ("SELECT id,original_title FROM dbo.tmdb_5000_movies_img where    len(overview) <= 0 or overview is null")


    # execute the query and read to a dataframe in Python
    df = pd.read_sql(query, cnxn)

    #print (data.describe());

    print (df)

    for index, row in df.iterrows():
        id = row['id']
        getAndUpdateImagePath(id)
        #print(row['id'], row['original_title'])


    del cnxn  # close the connection

test()
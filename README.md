# TabPy

## Install
```
pip install tabpy-server

pip install tornado==5.1.1

pip install -U scikit-learn

```

## Start TabPy Server
- Go to folder: \venv\Lib\site-packages\tabpy_server
- Run cmd: startup.bat

## Publish new function/method to TabPy server

Run the code below in Jupiter notebook or Pycharm notebook

```
# Connect to TabPy server using the client library
connection = tabpy_client.Client('http://localhost:9004/')

# define a method 
def Add(a,b):
    return (a+b)

# Publish the Add function to TabPy server so it can be used from Tableau
# Using the name TestAdd and a short description of what it does
connection.deploy('TestAdd',
                  Add,
                  'Returns a+b', override = True)

```

## Connect TabPy in Tableau
 - In Tableau Desktop, click the Help menu, and then select Settings and Performance > Manage External Service connection to open the External Service Connection dialog box:
 - Specify the type of analytics extension you want to connect to: RServe or TabPy/External API. The TabPy/External API option covers connections to TabPy and MATLAB.
 - Ref: https://help.tableau.com/v2020.1/pro/desktop/en-us/r_connection_manage.htm#configure-an-analytics-extensions-connection

## Create a calculated field and Python script in Tableau Desktop

Assuming that in the data source we have a table with two fields X, Y 
This script will return the sum of X and Y using method TestAdd above
```
SCRIPT_INT("
 print('aaa')
 print (_arg1) 
 print (_arg2)
 t = tabpy.query('TestAdd',_arg1,_arg2)['response']
 print ('sum') 
 print (t)
 print (sum(t))

 return sum(t)
",
SUM(X),SUM(Y))

```
## References
- https://www.tableau.com/about/blog/2018/8/working-external-services-tableau-tabpy-r-matlab-93351?_ga=2.222069278.748154300.1608156447-1598563553.1602561206
- https://www.tableau.com/about/blog/2017/1/building-advanced-analytics-applications-tabpy-64916
- https://www.tableau.com/about/blog/2016/11/leverage-power-python-tableau-tabpy-62077
- https://help.tableau.com/v2020.1/pro/desktop/en-us/r_connection_manage.htm#configure-an-analytics-extensions-connection
- https://help.tableau.com/current/server/en-us/config_r_tabpy.htm
- http://zachmoshe.com/2017/04/03/pickling-keras-models.html

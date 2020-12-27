# TabPy

## Install
```
pip install tabpy-server
```

## Start TabPy Server
- Go to folder: \venv\Lib\site-packages\tabpy_server
- Run cmd: startup.bat

## Connect TabPy in Tableau
 - In Tableau Desktop, click the Help menu, and then select Settings and Performance > Manage External Service connection to open the External Service Connection dialog box:
 - Specify the type of analytics extension you want to connect to: RServe or TabPy/External API. The TabPy/External API option covers connections to TabPy and MATLAB.
 - ...
https://help.tableau.com/v2020.1/pro/desktop/en-us/r_connection_manage.htm#configure-an-analytics-extensions-connection

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
##


cnxn_str = ("Driver={SQL Server Native Client 11.0};"
            "Server=BAOTRAN;"
            "Database=Python;"
           "Trusted_Connection=yes;")

date = datetime.today() - timedelta(days=7)  # get the date 7 days ago

date = date.strftime("%Y-%m-%d")  # convert to format yyyy-mm-dd

cnxn = pyodbc.connect(cnxn_str)  # initialise connection (assume we have already defined cnxn_str)

# build up our query string
query = ("SELECT * FROM Employee ")

# execute the query and read to a dataframe in Python
data = pd.read_sql(query, cnxn)

data;

del cnxn  # close the connection
import pymysql.cursors

class DBConnection:
    instance = None
    con = None

    def __new__(cls):
        if DBConnection.instance is None:
            DBConnection.instance = object.__new__(cls)
        return DBConnection.instance

    def __init__(self):
        if DBConnection.con is None:
            try:
                #Create Connecetions
                DBConnection.con = pymysql.connect(host='localhost',
                             user='root',
                             password='',
                             db='empsense',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)
                print('Database connection opened.')
            except pymysql.DatabaseError as db_error:
                print("Error :\n{0}".format(db_error))

    def insert_update(self, query):
        try:
            with DBConnection.con.cursor() as cursor:
                # Create a new record
                # sql = "INSERT INTO `users` (`email`, `password`) VALUES (%s, %s)"
                cursor.execute(query)
                # connection is not autocommit by default. So you must commit to save
                # your changes.
                DBConnection.con.commit()

        except pymysql.DatabaseError as db_error:
            print("Error :\n{0}".format(db_error))

    def read(self, query):
        try:
            with DBConnection.con.cursor() as cursor:
                # Read a single record
                cursor.execute(query)
                result = cursor.fetchone()
                #result = cursor.fetchall()
                print(result)

        except pymysql.DatabaseError as db_error:
            print("Error :\n{0}".format(db_error))

    def readDataSet(self, query):
        try:
            with DBConnection.con.cursor() as cursor:
                # Read a single record
                cursor.execute(query)
                result = cursor.fetchall()
                return result

        except pymysql.DatabaseError as db_error:
            print("Error :\n{0}".format(db_error))

    def __del__(self):
        if DBConnection.con is not None:
            DBConnection.con.close()
            print('Database connection closed.')


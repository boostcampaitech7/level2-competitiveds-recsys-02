import pymysql

class Mysql:
    def __init__(self, user):
        self.mysql = None
        self.cursor = None
        self.hostname = "10.28.224.161"
        self.port = 30111
        self.user = user
        self.password = "1234"
        self.db = "recsys_02"

    def db_connect(self):
        try:
            self.mysql = pymysql.connect(host = self.hostname, port = self.port, user = self.user, password = self.password, db = self.db)
            self.cursor = self.mysql.cursor(pymysql.cursors.DictCursor)
        except Exception as e:
            print(f"error : {e}")

    def db_disconnect(self):
        self.cursor.close()
        self.mysql.close()

    def db_insert(self, columns, values):
        columns_str = ', '.join(columns) 
        placeholders = ', '.join(['%s'] * len(values))
        
        try:
            self.cursor.execute(f"""
                INSERT INTO result 
                ({columns_str}) 
                VALUES ({placeholders})
            """, 
            tuple(values))  # Convert values list to a tuple
            self.mysql.commit()
            print("Your data has been saved successfully.")

        except Exception as e:
            print(f"error : {e}")

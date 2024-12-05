import pymysql

class Mysql:
    """
    MySQL 데이터베이스와의 연결 및 데이터를 삽입하는 클래스.

    Args:
        user (str): 데이터베이스에 연결할 사용자 이름.
    
    Methods:
        db_connect: MySQL 데이터베이스에 연결.
        db_disconnect: MySQL 데이터베이스 연결을 종료.
        db_insert: 특정 테이블에 데이터를 삽입.

    db_connect:
        - 데이터베이스에 연결하고 커서를 설정.
        - Exception: 연결 실패 시 오류 메시지 출력.

    db_disconnect:
        - 현재 활성화된 커서와 데이터베이스 연결을 닫음.

    db_insert:
        - 주어진 열과 값을 기반으로 'result' 테이블에 데이터를 삽입.
        - Args:
            columns (list): 삽입할 데이터의 열 이름 목록.
            values (list): 삽입할 데이터 값 목록.
        - Exception: 삽입 실패 시 오류 메시지 출력.
    """
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
            tuple(values))
            self.mysql.commit()
            print("Your data has been saved successfully.")

        except Exception as e:
            print(f"error : {e}")

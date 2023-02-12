
# create a table in sqlite3 database for pieces configuration
#     def create_table(self, table_name, columns):
#         self.conn.execute("CREATE TABLE IF NOT EXISTS " + table_name + " (" + columns + ")")  
#         self.conn.commit()
# self.conn.close()
# self.conn =  sqlite3.connect(db)
# self.cur = self.conn.cursor()

# write a function to create a table in sqlite3 database for pieces configuration
def create_table(self, table_name, columns):
    self.conn.execute("CREATE TABLE IF NOT EXISTS " + table_name + " (" + columns + ")")  
    self.conn.commit()
# self.conn.close()
import sqlite3
import timeit

# create database
conn = sqlite3.connect('database.db')
conn.execute('pragma journal_mode=wal')
cursor = conn.cursor()
sql = ('create table tb (date int , time text) ')
conn.execute(sql)
cursor.close()
conn.close()

# delete rows
conn = sqlite3.connect('database.db')
sql = 'delete from tb'
conn.execute(sql)
conn.commit()
conn.close()

# drop table
conn = sqlite3.connect('database.db')
sql = (f"drop table tb")
conn.execute(sql)
conn.commit()

# insert rows
conn = sqlite3.connect('database.db')
conn.execute('pragma journal_mode=wal')
start = timeit.default_timer()
for i in range(100):
    sql = (f"insert into tb values ({i}, 'hi {i}')")
    conn.execute(sql)
    conn.commit()
print("The difference of time is :", 
              timeit.default_timer() - start)

cursor = conn.cursor()
cursor.execute("select * from tb")
rows = cursor.fetchall()
for row in rows:
    print(row) 
cursor.close()
conn.close()


'''
always do cursor.close() as soon as possible after having done a (even read-only) query.
'''

# Situations Where SQLite Works Well
#    https://www.sqlite.org/whentouse.html
#    https://www.sqlitetutorial.net/


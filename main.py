import sqlite3
import time
import preprocess
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

print('Press [1] for Registration')
print('Press [2] for Attendance')
print('Press [3] to See the Attendance')
print('Press [4] for Downloading the Attendance')

user = int(input('Press any one : '))
if user == 1:
    name = input('Enter your Name : ')
    roll = input('Enter your Roll_number : ')
    designation = input('Enter your Designation : ')
    t0 = time.clock()
    face = preprocess.face_extract()
    encrypted_feature = preprocess.training(face)
    con = sqlite3.connect('feature_database.db', isolation_level=None)
    cur = con.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS features ( feature BLOB, name TEXT, roll_num TEXT, designation TEXT)")
    cur.execute("INSERT INTO features VALUES (?,?,?,?)", (encrypted_feature, name, roll, designation))
    con.commit()
    print('Data Saved')
    t1 = time.clock() - t0
    print('Time Taken -', t1)
elif user == 2:
    t0 = time.clock()
    face = preprocess.face_extract()
    encrypted_feature = preprocess.training(face)
    now = datetime.datetime.now()
    login_time = now.strftime("%H:%M:%S")
    con = sqlite3.connect('feature_database.db', isolation_level=None)
    cur = con.cursor()
    m = cur.execute("SELECT * FROM features")
    data_list = list()
    for i in m:
        data_list.append(i)
    con.close()
    for i in range(len(data_list)):
        fea = data_list[i][0]
        fea = np.frombuffer(fea, dtype='float32')
        fea = np.reshape(fea, (1, -1))
        sim = cosine_similarity(fea, encrypted_feature)
        print(sim)
        if sim.max() > .80:
            name = data_list[i][1]
            roll_number = data_list[i][2]
            designation = data_list[i][3]
            con = sqlite3.connect('database.db', isolation_level=None)
            cur = con.cursor()
            cur.execute('CREATE TABLE IF NOT EXISTS attendance (name TEXT,roll_number TEXT,designation TEXT,'
                        'Login_time TEXT)')
            cur.execute('INSERT INTO attendance VALUES (?,?,?,?)', (name, roll_number, designation, login_time))
            con.commit()
            con.close()
            print('Attendance successfully registered -', name)
            t1 = time.clock() - t0
            print('Time Taken -', t1)
            break
        else:
            print('NOT MATCHED')
            break

elif user == 3:
    con = sqlite3.connect('database.db', isolation_level=None)
    df = pd.read_sql_query("SELECT * FROM attendance", con)
    con.close()
    print(df)
elif user == 4:
    con = sqlite3.connect('database.db', isolation_level=None)
    df = pd.read_sql_query("SELECT * FROM attendance", con)
    con.close()
    df.to_csv('database.csv', index=False)
    print('Attendance file downloaded')
else:
    print('Service stopped')

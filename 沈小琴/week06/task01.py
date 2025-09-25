import sqlite3

# 连接到数据库，如果文件不存在会自动创建
conn = sqlite3.connect('student_management.db')
cursor = conn.cursor()

# 创建院系信息表
cursor.execute('''
DROP TABLE IF EXISTS departments;
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS departments (
    dept_id     INTEGER PRIMARY KEY,
    dept_name   TEXT NOT NULL,
    dean        TEXT
);
''')

# 创建学生信息表
cursor.execute('''
DROP TABLE IF EXISTS students;
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS students (
    student_id  TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    gender      TEXT CHECK(gender IN ('男', '女')),
    birth_date  TEXT,
    dept_id     INTEGER,
    FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
);
''')

# 创建课程信息表
cursor.execute('''
DROP TABLE IF EXISTS courses;
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS courses (
    course_id   TEXT PRIMARY KEY,
    course_name TEXT NOT NULL,
    credit      REAL CHECK(credit > 0),
    teacher     TEXT
);
''')

# 创建选课记录表
cursor.execute('''
DROP TABLE IF EXISTS enrollments;
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS enrollments (
    enrollment_id   INTEGER PRIMARY KEY,
    student_id      TEXT NOT NULL,
    course_id       TEXT NOT NULL,
    semester        TEXT,
    score           REAL CHECK(score BETWEEN 0 AND 100),
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    FOREIGN KEY (course_id) REFERENCES courses(course_id)
);
''')

# 提交更改
conn.commit()
print("数据库和表已成功创建。")

# 插入院系数据
cursor.execute("INSERT INTO departments (dept_name, dean) VALUES (?,?)", ('计算机学院', '张教授'))
cursor.execute("INSERT INTO departments (dept_name, dean) VALUES (?,?)", ('数学学院', '李教授'))
cursor.execute("INSERT INTO departments (dept_name, dean) VALUES (?,?)", ('物理学院', '王教授'))
conn.commit()
print("院系数据成功插入。")

# 插入学生数据
cursor.execute("INSERT INTO students (student_id, name, gender, birth_date, dept_id) VALUES (?,?,?,?,?)",
               ('20230001', '张三', '男', '2002-05-15', 1))
cursor.execute("INSERT INTO students (student_id, name, gender, birth_date, dept_id) VALUES (?,?,?,?,?)",
               ('20230002', '李四', '女', '2003-02-20', 1))
cursor.execute("INSERT INTO students (student_id, name, gender, birth_date, dept_id) VALUES (?,?,?,?,?)",
               ('20230003', '王五', '男', '2002-11-30', 2))
conn.commit()
print("学生数据成功插入。")

# 插入课程数据
cursor.execute("INSERT INTO courses (course_id, course_name, credit, teacher) VALUES (?,?,?,?)",
               ('CS101', '计算机基础', 3.0, '张教授'))
cursor.execute("INSERT INTO courses (course_id, course_name, credit, teacher) VALUES (?,?,?,?)",
               ('MA201', '高等数学', 4.0, '李教授'))
cursor.execute("INSERT INTO courses (course_id, course_name, credit, teacher) VALUES (?,?,?,?)",
               ('PH301', '大学物理', 3.5, '王教授'))
conn.commit()
print("课程数据成功插入。")

# 插入选课数据
cursor.execute("INSERT INTO enrollments (student_id, course_id, semester, score) VALUES (?,?,?,?)",
               ('20230001', 'CS101', '2023秋季', 92.5))
cursor.execute("INSERT INTO enrollments (student_id, course_id, semester, score) VALUES (?,?,?,?)",
               ('20230001', 'MA201', '2023秋季', 88.0))
cursor.execute("INSERT INTO enrollments (student_id, course_id, semester, score) VALUES (?,?,?,?)",
               ('20230002', 'CS101', '2023秋季', 95.0))
cursor.execute("INSERT INTO enrollments (student_id, course_id, semester, score) VALUES (?,?,?,?)",
               ('20230003', 'MA201', '2023秋季', 90.5))
conn.commit()
print("选课数据成功插入。")

# 查询学生及其院系信息
print("\n--- 学生及其院系信息 ---")
cursor.execute('''
SELECT students.student_id, students.name, students.gender, departments.dept_name
FROM students
JOIN departments ON students.dept_id = departments.dept_id;
''')

student_info = cursor.fetchall()
for student_id, name, gender, dept_name in student_info:
    print(f"学号: {student_id}, 姓名: {name}, 性别: {gender}, 院系: {dept_name}")

# 查询学生选课及成绩信息
print("\n--- 学生选课及成绩信息 ---")
cursor.execute('''
SELECT students.name, courses.course_name, enrollments.semester, enrollments.score
FROM enrollments
JOIN students ON enrollments.student_id = students.student_id
JOIN courses ON enrollments.course_id = courses.course_id;
''')

enrollment_info = cursor.fetchall()
for name, course_name, semester, score in enrollment_info:
    print(f"学生: {name}, 课程: {course_name}, 学期: {semester}, 成绩: {score}")

# 更新学生信息
print("\n--- 更新学生信息 ---")
cursor.execute("UPDATE students SET birth_date = ? WHERE student_id = ?", ('2002-06-10', '20230001'))
conn.commit()
print("学号20230001的学生出生日期已更新。")

# 查询更新后的学生信息
cursor.execute("SELECT student_id, name, birth_date FROM students WHERE student_id = '20230001'")
updated_student = cursor.fetchone()
print(f"更新后的信息: 学号: {updated_student[0]}, 姓名: {updated_student[1]}, 出生日期: {updated_student[2]}")

# 删除学生记录
print("\n--- 删除学生记录 ---")
cursor.execute("DELETE FROM students WHERE student_id = ?", ('20230003',))
conn.commit()
print("学号20230003的学生记录已被删除。")

# 查询剩余学生
print("\n--- 剩余学生信息 ---")
cursor.execute("SELECT student_id, name FROM students")
remaining_students = cursor.fetchall()
for student_id, name in remaining_students:
    print(f"学号: {student_id}, 姓名: {name}")

# 关闭连接
conn.close()
print("\n数据库连接已关闭。")
